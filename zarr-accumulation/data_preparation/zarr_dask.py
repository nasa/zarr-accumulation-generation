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
    if not block_info:
        return block

    # Trying with just unweighted block
    outputs = []
    for dim_i, dim_idx in enumerate(accumulation_dimensions):
        output = block.sum(axis=dim_idx)
        outputs.append(output.flatten())

    outputs = np.concatenate(outputs)

    # outputs = outputs.reshape(1, 1, len(outputs))
    # More generic than the above line for variable number of dimensions
    outputs = outputs.reshape(len(outputs))
    for i in range(len(block.shape) - 1):
        outputs = np.expand_dims(outputs, axis=i)
    # print("outputs.shape", outputs.shape)

    return outputs


def compute_write_zarr(
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
    idx_acc_time = int(batch_idx_start / variable_array_chunks[batch_dim_idx])
    print("idx_acc_time:", idx_acc_time)

    # Compute
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

        if len(chunk_indices) > 1:  # Don't reshape for acc_latlon
            # For dimension we're batching over (e.g., time)
            if dim_i == batch_dim_idx:
                result = result.reshape(array_shapes[dim_i][1], array_shapes[dim_i][2])
            else:
                result = result.reshape(
                    (
                        array_shapes[dim_i][0],
                        array_shapes[dim_i][1],
                        batch_idx_end - batch_idx_start,
                    )
                )

        # Loop over indices and perform cumsum in each dim
        # Skip cumsum for dimension we're batching over (e.g., dim "2" or time)
        if dim_i == batch_dim_idx:
            print("Skipping cumsum for batch dimension", batch_dim_idx)
        else:
            for i in range(len(dim_idx)):
                result = result.cumsum(axis=i)
            print("result.shape", result.shape)
            # print("result", result)

        # Convert result to less precise type to append
        result = result.astype("float32")

        # Save result to Zarr
        if dim_i == batch_dim_idx:
            dim_name = accumulation_names[dim_i] + "_temp"
            acc_group[dim_name][idx_acc_time, :, :] = result
        else:
            acc_group[accumulation_names[dim_i]][
                :, :, batch_idx_start:batch_idx_end
            ] = result

        # Update current_idx in flattened outputs
        current_idx = end_idx
        print("\n")
    return


def compute_batch_dimension(
    batch_array_dask,
    batch_dataset,
    batch_dim_stride,
    new_batch_array_chunks,
    batch_idx_start,
    batch_idx_end,
):
    # Do cumsum for the batch dimension (e.g., time) and update zarr array
    # Previously f_time()
    # The accumulation dimension is right now always the first dimension in the respective array, so axis=0
    result = (
        batch_array_dask[:, batch_idx_start:batch_idx_end, :]
        .cumsum(axis=0)[1::batch_dim_stride, :, :]
        .rechunk(new_batch_array_chunks)
        .astype("f4")
        .compute()
    )
    # print("result.shape", result.shape)
    # print("result", result)
    batch_dataset[:, batch_idx_start:batch_idx_end, :] = result

    # Operate on weights here

    return


def assemble_batch_dimension(
    batch_dim_idx,
    accumulation_names,
    accumulation_weight_names,
    acc_group,
    array_shapes,
    shape,
    variable_array_chunks,
):
    # Pick the first dimension (e.g., lat or 0) to batch over
    # These are hardcoded for now... make as parameters?
    batch_dim_idx_2 = 0
    n_threads = 9

    acc_dim_name = accumulation_names[batch_dim_idx]
    acc_dim_name_temp = acc_dim_name + "_temp"
    print(f"Final assembly for dimension: {acc_dim_name}")

    # Always a 1-element tuple e.g., (2,)
    batch_dim_stride = accumulation_strides[batch_dim_idx][0]

    # Create final dataset and run compute for the batch dimension (e.g., time)
    num_chunks_final = int(array_shapes[batch_dim_idx][0] / batch_dim_stride)
    chunk_size = variable_array_chunks[batch_dim_idx]
    num_batches = int(variable_array_chunks[batch_dim_idx_2] / n_threads)
    batch_size_per_thread = int(shape[batch_dim_idx_2] / n_threads)

    new_batch_array_shape = []
    new_batch_array_chunks = []
    # Putting the accumulation dim first
    for i in range(len(shape)):
        if i == batch_dim_idx:
            new_batch_array_shape.append(num_chunks_final)
            new_batch_array_chunks.append(int(variable_array_chunks[i]))

    # Other dims are in order after the accumulation dim
    for i in range(len(shape)):
        if i != batch_dim_idx:
            new_batch_array_shape.append(shape[i])
            if i == batch_dim_idx_2:
                new_batch_array_chunks.append(num_batches)
            else:
                new_batch_array_chunks.append(shape[i])

    print("new_batch_array_shape:", tuple(new_batch_array_shape))
    print("new_batch_array_chunks:", tuple(new_batch_array_chunks))

    batch_dataset = acc_group.create_dataset(
        acc_dim_name,
        shape=tuple(new_batch_array_shape),
        chunks=tuple(new_batch_array_chunks),
        compressor=compressor,
        # Filter goes here after figuring out how to handle it
        dtype="f4",
        overwrite=True,
    )

    # Process weights here

    # print(acc_group[acc_dim_name].shape)
    batch_array_dask = da.from_array(
        acc_group[acc_dim_name_temp], acc_group[acc_dim_name_temp].shape
    )
    print("batch_array_dask.shape", batch_array_dask.shape)
    # Weight array here

    processes = []
    for i in range(n_threads):
        batch_idx_start = i * batch_size_per_thread
        batch_idx_end = (i + 1) * batch_size_per_thread
        process = Process(
            target=compute_batch_dimension,
            args=(
                batch_array_dask,
                batch_dataset,
                batch_dim_stride,
                new_batch_array_chunks,
                batch_idx_start,
                batch_idx_end,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    # Delete temp dataset

    return


if __name__ == "__main__":
    # Setting variables - hardcoded for now.
    # We can perhaps let the user set these variables in command line arguments or in helper script
    batch_size = 100
    batch_dim_idx = 2  # np.argmax(shape) Turn this into the largest dimension, keeping time now to validate with old code

    # Open input Zarr store
    store_input = zarr.DirectoryStore("data/GPM_3IMERGHH_06_precipitationCal")
    root = zarr.open(store_input)
    variable_array = root["variable"]

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
        for dim_j, tup_val in enumerate(dim_idx_tuple):
            # print("tup value: ", tup_val)
            # print("(shape[tup]", shape[tup_val])
            # print("chunks[tup]", chunks[tup_val])
            # print("stride[tup]", stride[dim_j], "\n")

            # Batch dimension (e.g., time) will not have stride applied until final assembly
            if dim_i == batch_dim_idx:
                stride_value = 1
            else:
                stride_value = stride[dim_j]
            element = int(shape[tup_val] / (chunks[tup_val] * stride_value))
            array_shape.append(element)

            # Batch dimension (e.g., time) will be 1 in chunks
            if dim_i == batch_dim_idx:
                element = 1
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
    # Currently making new dataset with copying attribute files over. There might be better
    for dim_i, (dim, dim_weight) in enumerate(
        zip(accumulation_names, accumulation_weight_names)
    ):
        # Creating new dataset with copying the attributes over
        attributes_dim = acc_group[dim].attrs["_ARRAY_DIMENSIONS"]
        attributes_stride = acc_group[dim].attrs["_ACCUMULATION_STRIDE"]

        # For the batch dim (e.g., time and time weight), first create a temp dataset
        if dim_i == batch_dim_idx:
            dim += "_temp"
            dim_weight += "_temp"
            print("temp dim", dim)
            print("temp dim_weight", dim_weight)

        dataset = acc_group.create_dataset(
            dim,
            shape=array_shapes[dim_i],
            chunks=array_chunks[dim_i],
            compressor=compressor,
            # Filter goes here after figuring out how to handle it
            dtype="f4",
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
    strides = []
    for tup_i, tup in enumerate(accumulation_strides):
        # Batch dimension (e.g., time) - apply stride at assembly step, so stride=1 here
        if tup_i == batch_dim_idx:
            strides.append(1)
        else:
            strides.append(tup[0])
    strides = np.array(strides)[: len(chunks)]

    variable_array_chunks = strides * np.array(chunks)
    print("variable_array_chunks", variable_array_chunks, "\n")
    variable_array_dask = da.from_array(
        variable_array.astype("f8"), variable_array_chunks
    )

    # Compute
    batch_dim = idx_to_dim[batch_dim_idx]
    batch_dim_chunk_size = variable_array_chunks[batch_dim_idx]
    batch_dim_num_chunks = int(shape[batch_dim_idx] / batch_dim_chunk_size)
    print(
        "batch_dim_chunk_size:",
        batch_dim_chunk_size,
        "\nbatch_dim_num_chunks:",
        batch_dim_num_chunks,
        "\n",
    )

    # # Test on 1 batch
    # i = 0
    # batch_idx_start = i * batch_dim_chunk_size
    # batch_idx_end = (i + 1) * batch_dim_chunk_size
    # print("Range: ", batch_idx_start, batch_idx_end)
    # compute_write_zarr(
    #     acc_group,
    #     array_shapes,
    #     array_chunks,
    #     accumulation_names,
    #     accumulation_weight_names,
    #     accumulation_dimensions,
    #     variable_array_dask,
    #     variable_array_chunks,
    #     weight_dask,
    #     batch_dim_idx,
    #     batch_idx_start,
    #     batch_idx_end,
    # )

    # # Compute the batch dimension - batch_dim_idx (e.g., time)
    # assemble_batch_dimension(
    #     batch_dim_idx,
    #     accumulation_names,
    #     accumulation_weight_names,
    #     acc_group,
    #     array_shapes,
    #     shape,
    #     variable_array_chunks,
    # )
    # exit()

    # Run entire dataset
    for batch_start in range(0, batch_dim_num_chunks, batch_size):
        print("Batch: ", batch_start)
        processes = []
        batch_end = min(batch_start + batch_size, batch_dim_num_chunks)
        for i in range(batch_start, batch_end):
            batch_idx_start = i * batch_dim_chunk_size
            batch_idx_end = (i + 1) * batch_dim_chunk_size
            print("Range: ", batch_idx_start, batch_idx_end)
            process = Process(
                target=compute_write_zarr,
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

    # Final compute and assembly for the batch dimension - batch_dim_idx (e.g., time)
    assemble_batch_dimension(
        batch_dim_idx,
        accumulation_names,
        accumulation_weight_names,
        acc_group,
        array_shapes,
        shape,
        variable_array_chunks,
    )
