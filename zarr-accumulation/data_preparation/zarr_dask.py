import time
import s3fs
import zarr
import json
import numpy as np
import dask.array as da
from dask import compute
from numcodecs import Blosc
from multiprocessing import Process
from codec_filter import DeltaLat, DeltaLon, DeltaTime
from collections import OrderedDict
from itertools import combinations

compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
s3 = s3fs.S3FileSystem()


def compute_block_sum(
    block, accumulation_dimensions, block_info=None, weight_dask=None
):
    # print("block_info", block_info)
    if not block_info:
        return block

    # Need to generalize this weighing - compute the weights outside of this function
    # (s0, _, _) = block.shape
    # (i1, i2) = block_info[0]["array-location"][0]
    # print(s0, i1, i2)
    # mask = block >= 0
    # weights = mask * (weight_dask[i1:i2].reshape(s0, 1, 1))
    # block_weighted = block * weights
    # block = block_weighted

    # Trying with just unweighted block first
    outputs = []
    for dim_i, dim_idx in enumerate(accumulation_dimensions):
        # Skipping time for now
        if dim_i == 2:
            continue
        output = block.sum(axis=dim_idx)
        outputs.append(output.flatten())

    outputs = np.concatenate(outputs)
    outputs = outputs.reshape(1, 1, len(outputs))
    return outputs


def run_compute_write_zarr(
    acc_group,
    array_shapes,
    array_chunks,
    accumulation_names,
    accumulation_weight_names,
    accumulation_dimensions,
    variable_array_dask,
    variable_array_chunks,
    weight_dask,
    batch_dim_idx,
    batch_idx_start,
    batch_idx_end,
):
    # Compute
    # Equivalent is np.take() but this function creates a copy of array
    slice_list = [slice(None)] * variable_array_dask.ndim
    slice_list[batch_dim_idx] = slice(batch_idx_start, batch_idx_end)
    # print("slice_list: ", slice_list)
    variable_block = variable_array_dask[tuple(slice_list)]
    # Validate the generic tuple slicing - delete later
    np.allclose(
        variable_block, variable_array_dask[:, :, batch_idx_start:batch_idx_end]
    )

    block_sums = variable_block.map_blocks(
        compute_block_sum,
        accumulation_dimensions,
        weight_dask=weight_dask,
        chunks=variable_array_chunks,
    ).compute()
    print("block_sums.shape:", block_sums.shape)

    # Extract data
    current_idx = 0
    for dim_i, dim_idx in enumerate(accumulation_dimensions):
        # Skipping time for now
        if dim_i == 2:
            continue

        # Get idx of array in flattened outputs
        chunk_indices = np.arange(0, len(block_sums.shape))
        print("chunk_indices", chunk_indices)
        remove_mask = np.in1d(chunk_indices, np.array(dim_idx))
        print("remove_mask", remove_mask)
        chunk_indices = chunk_indices[~remove_mask]
        print("chunk_indices", chunk_indices)
        chunk_sizes_to_index = variable_array_chunks[chunk_indices]
        print("chunk_sizes_to_index", chunk_sizes_to_index)

        # Multiply chunk sizes in idx to get in flattened array
        end_idx = current_idx + np.prod(chunk_sizes_to_index)
        print("current_idx", current_idx)
        print("end_idx", end_idx)

        # Get sum result and compute cumsum
        result = block_sums[:, :, current_idx:end_idx]

        # Need to generalize this for other dims
        # Especially latlon - which doesn't need reshaping and will have 2 cumsum operations over 2 dimensions
        # Also transpose here to .transpose((0, 2, 1)) if figure out how to generalize this
        if len(chunk_indices) > 1:  # Don't reshape for acc_latlon
            result = result.reshape(
                (
                    array_shapes[dim_i][0],
                    array_shapes[dim_i][1],
                    batch_idx_end - batch_idx_start,
                )
            )

        # Loop over indices and perform cumsum in each dim
        for i in range(len(dim_idx)):
            # result.cumsum(axis=dim_idx[i]) -> wrong
            # print("i:", i)
            result = result.cumsum(axis=i)
        print("result.shape", result.shape)

        # Save result to Zarr
        acc_group[accumulation_names[dim_i]][
            :, :, batch_idx_start:batch_idx_end
        ] = result

        # Update current_idx in flattened outputs
        current_idx = end_idx

        print("\n")

    return


if __name__ == "__main__":
    # Open input Zarr store
    store_input = zarr.DirectoryStore("data/GPM_3IMERGHH_06_precipitationCal")
    root = zarr.open(store_input)
    variable_array = root["variable"]

    # Create accumulation group and the attribute file
    # Assume user has created this attribute file, so only doing here during developing code
    acc_group = root.create_group("variable_accumulation_group", overwrite=True)
    acc_group.attrs["_ACCUMULATION_GROUP"] = {
        "latitude": {
            "_DATA_WEIGHTED": "acc_lat",
            "_WEIGHTS": "acc_wt_lat",
            "longitude": {
                "_DATA_WEIGHTED": "acc_lat_lon",
                "_WEIGHTS": "acc_wt_lat_lon",
                "time": {},
            },
            "time": {},
        },
        "longitude": {
            "_DATA_WEIGHTED": "acc_lon",
            "_WEIGHTS": "acc_wt_lon",
            "time": {},
        },
        "time": {"_DATA_WEIGHTED": "acc_time", "_WEIGHTS": "acc_wt_time"},
    }

    # Get accumulation group
    acc_group = root["variable_accumulation_group"]

    # Assuming the attributes are pre-defined, read them in as a dictionary
    acc_attrs = OrderedDict(acc_group.attrs)
    dimensions = acc_attrs["_ACCUMULATION_GROUP"].keys()
    print("acc_attrs:", acc_attrs, "\n")

    # Construct dictionaries as mappings between dimension name and index
    dim_to_idx = {}
    for dim_i, dim in enumerate(dimensions):
        dim_to_idx[dim] = dim_i
    print("dim_to_idx:", dim_to_idx)
    # Swap key and value
    idx_to_dim = {v: k for k, v in dim_to_idx.items()}
    print("idx_to_dim:", idx_to_dim, "\n")

    # Contruct a combination list of the dimensions
    dim_combinations = []
    for i in range(1, len(dimensions) + 1):
        dim_combinations += list(combinations(dimensions, i))
    print("dim_combinations:", dim_combinations, "\n")

    # Construct spec lists of accumulation array names and the indices
    # Changed from dictionary to lists
    accumulation_names = []
    accumulation_weight_names = []
    accumulation_dimensions = []
    for tup in dim_combinations:
        # print(tup)
        temp_dim = acc_attrs["_ACCUMULATION_GROUP"]
        dim_indices = []
        for dim in tup:
            # print("dim", dim)
            temp_dim = temp_dim[dim]
            dim_indices.append(dim_to_idx[dim])

        if temp_dim:  # If dictionary is not empty for these dimensions
            # print("temp_dim", temp_dim)
            accumulation_names.append(temp_dim["_DATA_WEIGHTED"])
            accumulation_weight_names.append(temp_dim["_WEIGHTS"])
            accumulation_dimensions.append(tuple(dim_indices))

    print("accumulation_names:", accumulation_names)
    print("accumulation_weight_names:", accumulation_weight_names)
    print("accumulation_dimensions:", accumulation_dimensions)

    # Innitialize data arrays for accumulation datasets
    # Later remove this part - assume user creates datasets with zattr files?
    for dim in accumulation_names + accumulation_weight_names:
        dataset = acc_group.create_dataset(
            dim,
            shape=(0, 0, 0),  # Dummy empty shape, will update after getting stride info
            overwrite=True,
        )

        # Making the zattr for each dataset here while developing code
        if dim == "acc_lat" or dim == "acc_wt_lat":
            stride_list = [2, 0, 0]
        if dim == "acc_lon" or dim == "acc_wt_lon":
            stride_list = [0, 2, 0]
        if dim == "acc_lat_lon" or dim == "acc_wt_lat_lon":
            stride_list = [2, 2, 0]
        if dim == "acc_time" or dim == "acc_wt_time":
            stride_list = [0, 0, 1]
        dataset.attrs["_ARRAY_DIMENSIONS"] = ["latitude", "longitude", "time"]
        dataset.attrs["_ACCUMULATION_STRIDE"] = stride_list

    # Get stride info and create arrays for length/shape/chunks
    accumulation_strides = []
    for dim in accumulation_names:
        stride = tuple(
            [s for s in dict(acc_group[dim].attrs)["_ACCUMULATION_STRIDE"] if s != 0]
        )
        accumulation_strides.append(stride)
    print("accumulation_strides:", accumulation_strides, "\n")

    shape = variable_array.shape
    chunks = variable_array.chunks
    print("shape:", shape, "-- chunks:", chunks)

    array_shapes = []
    array_chunks = []
    for dim_i, dim_idx_tuple in enumerate(accumulation_dimensions):
        stride = accumulation_strides[dim_i]
        array_shape = []
        array_chunk = []
        for dim_j, tup in enumerate(dim_idx_tuple):
            element = int(shape[dim_j] / (chunks[dim_j] * stride[dim_j]))
            array_shape.append(element)
            array_chunk.append(element)

        for dim_k in range(len(shape)):
            if dim_k not in dim_idx_tuple:
                shape_element = shape[dim_k]
                chunk_element = chunks[dim_k]
                array_shape.append(shape_element)
                array_chunk.append(chunk_element)

        array_shapes.append(array_shape)
        array_chunks.append(array_chunk)
    print("array_shapes:", array_shapes)
    print("array_chunks:", array_chunks, "\n")

    # Update Zarr arrays' shape and chunks
    for dim_i, (dim, dim_weight) in enumerate(
        zip(accumulation_names, accumulation_weight_names)
    ):
        # This works to update shape: acc_group[dim].shape = array_shapes[dim_i]
        # But this line gives error: AttributeError: can't set attribute
        # "Changing the chunk size of a zarr array would require rewriting all of the data."
        # acc_group[dim].chunks = # array_chunks[dim_i]
        # So creating new dataset with copying the attributes over
        attributes_dim = acc_group[dim].attrs["_ARRAY_DIMENSIONS"]
        attributes_stride = acc_group[dim].attrs["_ACCUMULATION_STRIDE"]
        dataset = acc_group.create_dataset(
            dim,
            shape=array_shapes[dim_i],
            chunks=array_chunks[dim_i],
            compressor=compressor,
            # Filter goes here after figuring out how to handle it
            dtype="f4",  # Need to change for time
            overwrite=True,
        )
        dataset.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
        dataset.attrs["_ACCUMULATION_STRIDE"] = attributes_stride

        dataset_weight = acc_group.create_dataset(
            dim_weight,
            shape=array_shapes[dim_i],
            chunks=array_chunks[dim_i],
            compressor=compressor,
            # Filter goes here after figuring out how to handle it
            dtype="f4",  # Need to change for time
            overwrite=True,
        )
        dataset_weight.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
        dataset_weight.attrs["_ACCUMULATION_STRIDE"] = attributes_stride

    # Create weight array
    # How to generalize this? Here, assuming the first dimension (e.g., lat) has weights
    weight = np.cos(np.deg2rad([np.arange(-89.95, 90, 0.1)]))[:, : shape[0]].reshape(
        shape[0], 1, 1
    )
    print("weight.shape:", weight.shape)
    weight_dask = da.from_array(weight)

    # Convert data to dask array
    # Update chunks after applying strides
    strides = np.array([tup[0] for tup in accumulation_strides])[: len(chunks)]
    variable_array_chunks = strides * np.array(chunks)
    print("variable_array_chunks", variable_array_chunks, "\n")
    variable_array_dask = da.from_array(
        variable_array.astype("f8"), variable_array_chunks
    )

    ##########
    # import copy

    # # Make weighted variable array
    # weight_dim_idx = 0
    # weight_dim = idx_to_dim[weight_dim_idx]  # latitude
    # x = copy.deepcopy(variable_array_dask) >= 0
    # weighted_array =
    # print(x)
    # exit()

    # Compute
    # How to generalize the batches - pick an arbitrary dimension to use?
    # Need a place to set these variables - how is this batch size chosen?
    batch_size = 100
    batch_dim_idx = 2
    batch_dim = idx_to_dim[batch_dim_idx]  # time
    batch_dim_chunk_size = variable_array_chunks[batch_dim_idx]
    batch_dim_num_chunks = int(shape[batch_dim_idx] / batch_dim_chunk_size)
    print(
        "batch_dim_chunk_size:",
        batch_dim_chunk_size,
        "batch_dim_num_chunks:",
        batch_dim_num_chunks,
        "\n",
    )

    # Test on 1 batch
    i = 0
    batch_idx_start = i * batch_dim_chunk_size
    batch_idx_end = (i + 1) * batch_dim_chunk_size
    print("Range: ", batch_idx_start, batch_idx_end)
    run_compute_write_zarr(
        acc_group,
        array_shapes,
        array_chunks,
        accumulation_names,
        accumulation_weight_names,
        accumulation_dimensions,
        variable_array_dask,
        variable_array_chunks,
        weight_dask,
        batch_dim_idx,
        batch_idx_start,
        batch_idx_end,
    )
    exit()

    for batch_start in range(0, batch_dim_num_chunks, batch_size):
        print("Batch: ", batch_start)
        processes = []
        batch_end = min(batch_start + batch_size, batch_dim_num_chunks)
        for i in range(batch_start, batch_end):
            batch_idx_start = i * batch_dim_chunk_size
            batch_idx_end = (i + 1) * batch_dim_chunk_size
            print("Range: ", batch_idx_start, batch_idx_end)
            process = Process(
                target=run_compute_write_zarr,
                args=(
                    acc_group,
                    array_shapes,
                    array_chunks,
                    accumulation_names,
                    accumulation_weight_names,
                    accumulation_dimensions,
                    variable_array_dask,
                    variable_array_chunks,
                    weight_dask,
                    batch_dim_idx,
                    batch_idx_start,
                    batch_idx_end,
                ),
            )
            process.start()
            processes.append(process)

            # exit()

        for process in processes:
            process.join()

