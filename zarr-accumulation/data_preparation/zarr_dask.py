import time
import s3fs
import zarr
import json
import numpy as np
import dask.array as da
from dask import compute
from numcodecs import Blosc
from multiprocessing import Process
from codec_filter import AccumulationDeltaFilter
from collections import OrderedDict
from itertools import combinations

compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
s3 = s3fs.S3FileSystem()


def weight_computation_function(
    block, block_info, fill_value, dimension_arrays_dict, weight_dimension="latitude"
):
    (s0, _, _) = block.shape
    (i1, i2) = block_info[0]["array-location"][0]

    # weight_dimension should be user-specified in main?
    dimension_array = dimension_arrays_dict[weight_dimension]
    block_dimension_array = dimension_array[i1:i2]

    # The way to compute weights should also be user-specified?
    block_weights = np.cos(np.deg2rad(block_dimension_array))

    mask = block >= fill_value
    weights = mask * (block_weights.reshape(s0, 1, 1))
    block_weighted = block * weights

    return weights, block_weighted


def compute_block_sum(
    block,
    fill_value,
    accumulation_dimensions,
    block_info=None,
    dimension_arrays_dict=None,
):

    if not block_info:
        return block

    # weight_computation_function to get weights and weighted block
    weights, block_weighted = weight_computation_function(
        block, block_info, fill_value, dimension_arrays_dict
    )

    # Data sums
    outputs = []
    for dim_i, dim_idx in enumerate(accumulation_dimensions):
        # output = block.sum(axis=dim_idx) # Unweighted block
        output = block_weighted.sum(axis=dim_idx)
        outputs.append(output.flatten())

    # Weight sums - probably will turn this into a function to not repeat this code
    # outputs_weights = []
    for dim_i, dim_idx in enumerate(accumulation_dimensions):
        output_weight = weights.sum(axis=dim_idx)
        outputs.append(output_weight.flatten())

    outputs = np.concatenate(outputs)
    # outputs = outputs.reshape(1, 1, len(outputs))
    # More generic than the above line for variable number of dimensions
    outputs = outputs.reshape(len(outputs))
    for i in range(len(block.shape) - 1):
        outputs = np.expand_dims(outputs, axis=i)

    # print("outputs.shape, block_info", outputs.shape, block_info)
    return outputs


def extract_data(
    block_sums,
    acc_group,
    accumulation_names,
    accumulation_dimensions,
    batch_dim_idx,
    accumulation_dim_orders_idx,
    variable_array_chunks,
    batch_idx_start,
    batch_idx_end,
    idx_acc,
    current_idx,
):
    # Extract data
    # current_idx = 0
    for dim_i, dim_idx in enumerate(accumulation_dimensions):

        # print(f"\nDim: {accumulation_names[dim_i]}")

        # Get idx of array in flattened outputs
        chunk_indices = np.arange(0, len(block_sums.shape))
        # print("chunk_indices", chunk_indices)
        remove_mask = np.in1d(chunk_indices, np.array(dim_idx))
        # print("remove_mask", remove_mask)
        chunk_indices = chunk_indices[~remove_mask]
        # print("chunk_indices", chunk_indices)
        chunk_sizes_to_index = variable_array_chunks[chunk_indices]
        # print("chunk_sizes_to_index", chunk_sizes_to_index)

        # Multiply chunk sizes in idx to get in flattened array
        end_idx = current_idx + np.prod(chunk_sizes_to_index)
        # print("current_idx", current_idx)
        # print("end_idx", end_idx)

        # Get sum result and compute cumsum
        result = block_sums[:, :, current_idx:end_idx]
        # print("data[:, :, idx_1:idx_2]", result.mean())

        ##### New change to adapt to user's defined final dimension order
        result_shape = result.shape
        # print("result.shape:", result.shape)  # lat: (25, 25, 28800)
        reshape_1_shapes = []
        reshape_1_indices = []
        # Loop over num chunk dimensions besides the last (e.g., nalat, nalon)
        for i, shape in enumerate(result_shape[:-1]):
            reshape_1_shapes.append(shape)
            reshape_1_indices.append(i)
        # print("reshape_1_shapes", reshape_1_shapes)  # [25, 25]
        # print("reshape_1_indices", reshape_1_indices)  # [0, 1]

        # Add to the lists the chunk dimensions.
        # For example, for lat: (e.g., clat, clon).
        # For the first reshape we have the shapes e.g., (nalat, nalon, clon, ctime)
        # and the corresponding indices e.g. (0, 1, 1, 2)
        reshape_1_shapes.extend(list(chunk_sizes_to_index))
        reshape_1_indices.extend(list(chunk_indices))
        # print("reshape_1_shapes", reshape_1_shapes)  # lat: [25, 25, 144, 200]
        # print("reshape_1_indices", reshape_1_indices)  # lat: [0, 1, 1, 2]

        # Do first reshape
        result = result.reshape(reshape_1_shapes)
        # print("result.shape", result.shape)
        # print("reshape 1", result.mean())

        # Transpose to bring together the same dims e.g., (nalat, nalon, clon, ctime) if they're not adjacent already
        # For lon, we'd go from (nalat, nalon, clat, ctime) -> (nalat, clat, nalon, ctime)
        # Get the indices that would sort reshape_1_indices
        sort_indices = np.argsort(reshape_1_indices)
        # print("sort_indices", sort_indices)  # lat: [0 1 2 3]
        new_indices = np.sort(reshape_1_indices)  # sort this to use later
        # print("new_indices", new_indices)

        # Apply sort_indices to result.transpose()
        # result = result.transpose(tuple(sort_indices))
        # print("result.shape", result.shape)  # lat: (25, 25, 144, 200)

        result = result.transpose(tuple(sort_indices))
        # print("result.shape", result.shape)  # lat: (25, 25, 144, 200)
        # print("transpose 1", result.mean())

        # Second reshape operation to multiply the dim that appears more than once
        # E.g., lat: (nalat, nalon * clon, ctime), lon: (nalat * clat, nalon, ctime)
        # First, find duplicates in reshape_1_indices
        seen_indices = []  # set()
        reshape_2_shapes = []
        for idx, value in enumerate(new_indices):
            if value not in seen_indices:
                # If unique value
                reshape_2_shapes.append(result.shape[idx])
                # seen_indices.add(value)
                seen_indices.append(value)
                # print(reshape_2_shapes, seen_indices)
            else:
                # If duplicate value, multiply the corresponing element in
                # result.shape with with previous element in reshape_2_shapes (i.e., reshape_2_shapes[-1])
                # to update this value in reshape_2_shapes
                # FIX ^: multiply to idx thats the same not the previous element
                # print(idx, value)
                # print(np.where(seen_indices == value)[0][0])
                reshape_2_shapes[np.where(seen_indices == value)[0][0]] *= result.shape[
                    idx
                ]
                # reshape_2_shapes[-1] *= result.shape[idx]
                # reshape_2_shapes[-1] *= result.shape[idx]
        # print("reshape_2_shapes", reshape_2_shapes)  # Lat: [25, 3600, 200]
        # Apply the second reshape to result
        result = result.reshape(reshape_2_shapes)
        # print("result.shape", result.shape)  # lat: (25, 3600, 200)
        # print("reshape 2", result.mean())

        # Compute cumsum over the accumulation dim (dim_idx)
        # Skip cumsum for batch dim
        if dim_i != batch_dim_idx:
            for i in dim_idx:
                # print("dim_idx - cumsum axis", i)
                result = result.cumsum(axis=i)
            # print("result.shape", result.shape)  # lat: (25, 3600, 200)
            # print("cumsum", result.mean())

        # Final transpose to user-specified order, except for batch dim (e.g., time)
        if dim_i != batch_dim_idx:
            result = result.transpose(accumulation_dim_orders_idx[dim_i])
            # print("result.shape", result.shape)  # lat: (25, 200, 3600)
            # print("transpose 2", result.mean())

        # Save result to Zarr
        assigning_slice = []
        for i, dim_order_idx in enumerate(accumulation_dim_orders_idx[dim_i]):
            # print(dim_order_idx)
            if dim_order_idx == batch_dim_idx:
                # print(dim_order_idx)
                if dim_i != batch_dim_idx:
                    assigning_slice.append(slice(batch_idx_start, batch_idx_end, None))
                else:
                    # For the batch dim, use idx_acc, not batch_idx_start:batch_idx_end
                    # start_slice_idx = np.max(0, idx_acc - 1)  # To avoid negative
                    assigning_slice.append(slice(idx_acc, idx_acc + 1, None))

                    # Expand a dimension for result at this idx
                    result = np.expand_dims(result, axis=i)
                    # print("result.shape - expand dim", result.shape)
            else:
                # Equivalent to : (full slice)
                assigning_slice.append(slice(None, None, None))
        # print(assigning_slice)

        dim_name = accumulation_names[dim_i]
        if dim_i == batch_dim_idx:
            dim_name += "_temp"
        # print(
        #     "acc_group[dim_name][tuple(assigning_slice)].shape",
        #     acc_group[dim_name][tuple(assigning_slice)].shape,
        # )
        acc_group[dim_name][tuple(assigning_slice)] = result

        # Update current_idx in flattened outputs
        current_idx = end_idx
        # print("\n")

    return current_idx


def compute_write_zarr(
    acc_group,
    array_shapes,
    array_chunks,
    accumulation_names,
    accumulation_weight_names,
    accumulation_dimensions,
    accumulation_dim_orders_idx,
    variable_array_dask,
    variable_array_chunks,
    batch_dim_idx,
    batch_idx_start,
    batch_idx_end,
    dimension_arrays_dict,
    fill_value,
):

    idx_acc = int(batch_idx_start / variable_array_chunks[batch_dim_idx])
    # print("idx_acc:", idx_acc)

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
        fill_value,
        accumulation_dimensions,
        chunks=variable_array_chunks,
        dimension_arrays_dict=dimension_arrays_dict,
    ).compute()
    # print("block_sums.shape:", block_sums.shape)

    # Extract data for data and weights
    current_idx = extract_data(
        block_sums,
        acc_group,
        accumulation_names,
        accumulation_dimensions,
        batch_dim_idx,
        accumulation_dim_orders_idx,
        variable_array_chunks,
        batch_idx_start,
        batch_idx_end,
        idx_acc,
        current_idx=0,
    )
    # print("current_idx", current_idx)

    current_idx = extract_data(
        block_sums,
        acc_group,
        accumulation_weight_names,
        accumulation_dimensions,
        batch_dim_idx,
        accumulation_dim_orders_idx,
        variable_array_chunks,
        batch_idx_start,
        batch_idx_end,
        idx_acc,
        current_idx=current_idx,
    )

    return


def compute_batch_dimension(
    batch_array_dask,
    batch_array_dask_weight,
    batch_dataset,
    batch_dataset_weight,
    dim_order_idx,
    batch_dim_idx,
    batch_dim_idx_2,
    batch_dim_stride,
    new_batch_array_chunks,
    batch_idx_start,
    batch_idx_end,
    data_type,
):
    # print("batch_array_dask.shape", batch_array_dask.shape)

    # Do cumsum for the batch dimension (e.g., time) and update zarr array
    # Previously f_time()
    slice_list = [slice(None)] * batch_array_dask.ndim
    slice_list[batch_dim_idx_2] = slice(batch_idx_start, batch_idx_end)
    # print(batch_dim_idx_2, slice_list)
    # print(slice_list)

    cumsum_axis = list(dim_order_idx).index(batch_dim_idx)
    # print(cumsum_axis)
    slice_list_cumsum = [slice(None)] * batch_array_dask.ndim
    slice_list_cumsum[cumsum_axis] = slice(1, None, batch_dim_stride)
    # print(slice_list_cumsum)

    result = (
        batch_array_dask[tuple(slice_list)]
        .cumsum(axis=cumsum_axis)[tuple(slice_list_cumsum)]  # No transpose needed
        .rechunk(new_batch_array_chunks)
        .astype(data_type)
        .compute()
    )
    # print("result.shape", result.shape)
    # print("result", result)
    batch_dataset[tuple(slice_list)] = result

    # Operate on weights here
    result_weight = (
        batch_array_dask_weight[tuple(slice_list)]
        .cumsum(axis=cumsum_axis)[tuple(slice_list_cumsum)]  # No transpose needed
        .rechunk(new_batch_array_chunks)
        .astype(data_type)
        .compute()
    )
    # print("result.shape", result.shape)
    # print("result", result)
    batch_dataset_weight[tuple(slice_list)] = result_weight

    return


def assemble_batch_dimension(
    batch_dim_idx,
    accumulation_names,
    accumulation_weight_names,
    accumulation_dim_orders_idx,
    acc_group,
    array_shapes,
    shape,
    num_chunks,
    variable_array_chunks,
    data_type,
    batch_dim_idx_2=0,
    n_threads=9,
):

    # Pick the first dimension (e.g., lat or 0) to batch over
    # batch_dim_idx_2 and n_threads can perhaps become user-selected parameters

    acc_dim_name = accumulation_names[batch_dim_idx]
    acc_dim_name_temp = acc_dim_name + "_temp"
    acc_dim_name_weight = accumulation_weight_names[batch_dim_idx]
    acc_dim_name_temp_weight = acc_dim_name_weight + "_temp"
    print(f"Final assembly for dimension: {acc_dim_name} & {acc_dim_name_temp_weight}")

    attributes_dim = acc_group[acc_dim_name].attrs["_ARRAY_DIMENSIONS"]
    attributes_stride = acc_group[acc_dim_name].attrs["_ACCUMULATION_STRIDE"]
    # print(attributes_dim, attributes_stride)

    # Always a 1-element tuple e.g., (2,)
    batch_dim_stride = accumulation_strides[batch_dim_idx][0]
    # print("batch_dim_stride", batch_dim_stride)

    # Create final dataset and run compute for the batch dimension (e.g., time)
    # num_chunks_final = int(array_shapes[batch_dim_idx][0] / batch_dim_stride)
    num_chunks_final = int(num_chunks[batch_dim_idx] / batch_dim_stride)
    chunk_size = variable_array_chunks[batch_dim_idx]
    num_batches = int(variable_array_chunks[batch_dim_idx_2] / n_threads)
    batch_size_per_thread = int(shape[batch_dim_idx_2] / n_threads)

    # print("num_chunks_final:", num_chunks_final)
    # print(
    #     "variable_array_chunks[batch_dim_idx_2]", variable_array_chunks[batch_dim_idx_2]
    # )  # 72
    # print("num_batches:", num_batches)  # 8
    # print("batch_size_per_thread:", batch_size_per_thread)  # 200

    new_batch_array_shape = []
    new_batch_array_chunks = []
    # Make final shape and chunks for this dim
    dim_idx_tuple = accumulation_dimensions[batch_dim_idx]  # (2,)
    dim_order_idx = accumulation_dim_orders_idx[batch_dim_idx]  # (0, 2, 1)
    for dim_j, idx_val in enumerate(dim_order_idx):
        if idx_val in dim_idx_tuple:
            # Append number of chunks in the accumulation dimension
            new_batch_array_shape.append(num_chunks_final)
            new_batch_array_chunks.append(int(variable_array_chunks[idx_val]))
        elif idx_val == batch_dim_idx_2:
            new_batch_array_shape.append(shape[idx_val])
            new_batch_array_chunks.append(num_batches)
        else:
            new_batch_array_shape.append(shape[idx_val])
            new_batch_array_chunks.append(shape[idx_val])
    # print("new_batch_array_shape:", new_batch_array_shape)
    # print("array_chunk:", new_batch_array_chunks, "\n")

    batch_dataset = acc_group.create_dataset(
        acc_dim_name,
        shape=tuple(new_batch_array_shape),
        chunks=tuple(new_batch_array_chunks),
        compressor=compressor,
        dtype=data_type,
        # Filter goes here after figuring out how to handle it
        #filters=[AccumulationDeltaFilter(accumulation_dimensions[batch_dim_idx], accumulation_strides[batch_dim_idx], accumulation_dim_orders_idx[batch_dim_idx])],
        overwrite=True,
    )
    # Copy attribute file to new dataset
    batch_dataset.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
    batch_dataset.attrs["_ACCUMULATION_STRIDE"] = attributes_stride

    # The same for weights
    batch_dataset_weight = acc_group.create_dataset(
        acc_dim_name_weight,
        shape=tuple(new_batch_array_shape),
        chunks=tuple(new_batch_array_chunks),
        compressor=compressor,
        # Filter goes here after figuring out how to handle it
        #filters=[AccumulationDeltaFilter(accumulation_dimensions[batch_dim_idx], accumulation_strides[batch_dim_idx], accumulation_dim_orders_idx[batch_dim_idx])],
        dtype=data_type,
        overwrite=True,
    )
    # Copy attribute file to new dataset
    batch_dataset_weight.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
    batch_dataset_weight.attrs["_ACCUMULATION_STRIDE"] = attributes_stride

    # Data array
    batch_array_dask = da.from_array(
        acc_group[acc_dim_name_temp], acc_group[acc_dim_name_temp].shape
    )
    # print("batch_array_dask.shape", batch_array_dask.shape)

    # Weight array
    batch_array_dask_weight = da.from_array(
        acc_group[acc_dim_name_temp_weight], acc_group[acc_dim_name_temp_weight].shape
    )

    processes = []

    # t = time.time()
    for i in range(n_threads):
        batch_idx_start = i * batch_size_per_thread
        batch_idx_end = (i + 1) * batch_size_per_thread
        # print("batch_idx_start, batch_idx_end", batch_idx_start, batch_idx_end)
        process = Process(
            target=compute_batch_dimension,
            args=(
                batch_array_dask,
                batch_array_dask_weight,
                batch_dataset,
                batch_dataset_weight,
                dim_order_idx,
                batch_dim_idx,
                batch_dim_idx_2,
                batch_dim_stride,
                new_batch_array_chunks,
                batch_idx_start,
                batch_idx_end,
                data_type,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    # print("compute_batch_dimension took: ", time.time() - t)

    # Delete temp dataset here?

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

    # Get the original data type
    data_type = variable_array.dtype.char + str(variable_array.dtype.itemsize)
    print("data_type:", data_type)

    # Get fill value
    fill_value = variable_array.fill_value

    # Get dimension arrays to eventually pass into weight_computation_function()
    dimension_arrays = sorted(root.arrays())
    for array_i, array in enumerate(dimension_arrays):
        if array[0] == "variable":
            # Already got variable above, so remove from this list
            dimension_arrays.pop(array_i)
    dimension_arrays_dict = dict(dimension_arrays)
    print("dimension_arrays_dict:", dimension_arrays_dict)

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
    accumulation_dim_orders = []
    accumulation_dim_orders_idx = []
    for dim in accumulation_names:
        stride = tuple(
            [s for s in dict(acc_group[dim].attrs)["_ACCUMULATION_STRIDE"] if s != 0]
        )
        accumulation_strides.append(stride)

        dim_order = []
        dim_order_idx = []
        for o in dict(acc_group[dim].attrs)["_ARRAY_DIMENSIONS"]:
            dim_order.append(o)
            dim_order_idx.append(dim_to_idx[o])
        # dim_order = tuple(
        #     [s for s in dict(acc_group[dim].attrs)["_ARRAY_DIMENSIONS"] if s != 0]
        # )
        accumulation_dim_orders.append(tuple(dim_order))
        accumulation_dim_orders_idx.append(tuple(dim_order_idx))
    print("accumulation_strides:", accumulation_strides)
    print("accumulation_dim_orders:", accumulation_dim_orders)
    print("accumulation_dim_orders_idx:", accumulation_dim_orders_idx, "\n")

    shape = variable_array.shape
    chunks = variable_array.chunks

    # Adapting these lists to the dim order the user has in dataset attribute files
    array_shapes = []
    array_chunks = []
    for dim_i, dim_idx_tuple in enumerate(accumulation_dimensions):
        # print("dim_idx_tuple:", dim_idx_tuple)
        stride = accumulation_strides[dim_i]
        dim_order_idx = accumulation_dim_orders_idx[dim_i]
        # print("stride:", stride)
        # print("dim_order_idx:", dim_order_idx)

        array_shape = []
        array_chunk = []
        count = 0
        for dim_j, idx_val in enumerate(dim_order_idx):
            # print("idx value: ", idx_val)
            if idx_val in dim_idx_tuple:
                # Append number of chunks in the accumulation dimension
                if dim_i == batch_dim_idx:
                    stride_value = 1
                else:
                    stride_value = stride[count]
                num_chunks = int(shape[idx_val] / (chunks[idx_val] * stride_value))
                array_shape.append(num_chunks)

                # Batch dimension (e.g., time) will be 1 in chunks at first
                if dim_i == batch_dim_idx:
                    num_chunks = 1
                array_chunk.append(num_chunks)
                count += 1
            else:
                # If not the accumulation dimension, then append the length and chunk size
                array_shape.append(shape[idx_val])
                array_chunk.append(chunks[idx_val])
        array_shapes.append(array_shape)
        array_chunks.append(array_chunk)
        # print("\n")
    print("array_shapes:", array_shapes)
    print("array_chunks:", array_chunks, "\n")

    # Update Zarr arrays' shape and chunks
    # Currently making new dataset with copying attribute files over
    for dim_i, (dim, dim_weight) in enumerate(
        zip(accumulation_names, accumulation_weight_names)
    ):
        # Creating new dataset with copying the attributes over
        attributes_dim = acc_group[dim].attrs["_ARRAY_DIMENSIONS"]
        attributes_stride = acc_group[dim].attrs["_ACCUMULATION_STRIDE"]

        arr_shape = array_shapes[dim_i]
        arr_chunks = array_chunks[dim_i]
        # print(arr_shape, arr_chunks)

        # For the batch dim (e.g., time and time weight), first create a temp dataset
        if dim_i == batch_dim_idx:
            dim += "_temp"
            dim_weight += "_temp"
            # print("temp dim", dim)
            # print("temp dim_weight", dim_weight)

        dataset = acc_group.create_dataset(
            dim,
            shape=arr_shape,
            chunks=arr_chunks,
            compressor=compressor,
            dtype=data_type,
            # Filter goes here after figuring out how to handle it
            #filters=[AccumulationDeltaFilter(accumulation_dimensions[dim_i], accumulation_strides[dim_i], accumulation_dim_orders_idx[dim_i])],
            overwrite=True,
        )
        dataset.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
        dataset.attrs["_ACCUMULATION_STRIDE"] = attributes_stride

        dataset_weight = acc_group.create_dataset(
            dim_weight,
            shape=arr_shape,
            chunks=arr_chunks,
            compressor=compressor,
            dtype=data_type,
            # Filter goes here after figuring out how to handle it
            # May not be needed for weights
            overwrite=True,
        )
        dataset_weight.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
        dataset_weight.attrs["_ACCUMULATION_STRIDE"] = attributes_stride

    # Convert data to dask array
    # Update chunks after applying strides
    strides = []
    for tup_i, tup in enumerate(accumulation_strides):
        # Batch dimension (e.g., time) - apply stride at assembly step, so stride=1 here
        if tup_i == batch_dim_idx:
            strides.append(1)
        else:
            strides.append(tup[0])
    strides = tuple(np.array(strides)[: len(chunks)])
    # print("strides:", strides)

    variable_array_chunks = strides * np.array(chunks)
    print("variable_array_chunks", variable_array_chunks, "\n")
    variable_array_dask = da.from_array(
        variable_array.astype("f8"), variable_array_chunks
    )

    # Get number of chunks
    num_chunks = tuple(np.array(shape) / variable_array_chunks)
    print(
        f"shape: {shape} -- strides: {strides} -- variable_array_chunks: {tuple(variable_array_chunks)}  -- num_chunks: {num_chunks}\n"
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

    # Test on 1 batch
    # t = time.time()
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
    #     accumulation_dim_orders_idx,
    #     variable_array_dask,
    #     variable_array_chunks,
    #     batch_dim_idx,
    #     batch_idx_start,
    #     batch_idx_end,
    #     dimension_arrays_dict,
    #     fill_value,
    # )
    # print("main compute used: ", time.time() - t)

    # t = time.time()
    # # Compute the batch dimension - batch_dim_idx (e.g., time)
    # assemble_batch_dimension(
    #     batch_dim_idx,
    #     accumulation_names,
    #     accumulation_weight_names,
    #     accumulation_dim_orders_idx,
    #     acc_group,
    #     array_shapes,
    #     shape,
    #     num_chunks,
    #     variable_array_chunks,
    #     data_type,
    # )
    # print("time compute used: ", time.time() - t)
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
                    accumulation_dim_orders_idx,
                    variable_array_dask,
                    variable_array_chunks,
                    batch_dim_idx,
                    batch_idx_start,
                    batch_idx_end,
                    dimension_arrays_dict,
                    fill_value,
                ),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    # Final compute and assembly for the batch dimension - batch_dim_idx (e.g., time)
    assemble_batch_dimension(
        batch_dim_idx,
        accumulation_names,
        accumulation_weight_names,
        accumulation_dim_orders_idx,
        acc_group,
        array_shapes,
        shape,
        num_chunks,
        variable_array_chunks,
        data_type,
    )

