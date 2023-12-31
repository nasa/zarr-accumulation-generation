import numpy as np
import dask.array as da
from numcodecs import Blosc
import utilities.compute as compute_functions


def extract_data(
    data_info_dict,
    block_sums,
    batch_dim_idx,
    batch_idx_start,
    batch_idx_end,
    idx_acc,
    current_idx,
    weights_flag,
):
    """
    Extract and update accumulation data in Zarr arrays based on block_sums.

    This function extracts computed accumulation data from block_sums and updates corresponding Zarr arrays.
    The extracted data is reshaped and transposed to match the required dimensions.

    Args:
        data_info_dict (dict): A dictionary containing various information about the data and accumulations.
        block_sums (numpy.ndarray): An array containing computed sums for data blocks.
        batch_dim_idx (int): Index of the batch dimension.
        batch_idx_start (int): Start index for the batch dimension slice.
        batch_idx_end (int): End index for the batch dimension slice.
        idx_acc (int): Accumulation index to update in the Zarr arrays.
        current_idx (int): Current index in flattened outputs.
        weights_flag (bool): Flag indicating whether the weights or the data are being processed.

    Returns:
        int: The updated current_idx in flattened outputs.

    """

    for dim_i, dim_idx in enumerate(data_info_dict["accumulation_dimensions"]):
        # Get idx of array in flattened outputs
        chunk_indices = np.arange(0, len(block_sums.shape))
        remove_mask = np.in1d(chunk_indices, np.array(dim_idx))
        chunk_indices = chunk_indices[~remove_mask]
        chunk_sizes_to_index = data_info_dict["variable_array_chunks"][chunk_indices]

        # Multiply chunk sizes in idx to get in flattened array
        end_idx = current_idx + np.prod(chunk_sizes_to_index)

        # Get sum result and compute cumsum
        result = block_sums[:, :, current_idx:end_idx]

        # Adapt to user's defined final dimension order
        result_shape = result.shape
        reshape_1_shapes = []
        reshape_1_indices = []
        # Loop over num chunk dimensions besides the last (e.g., nalat, nalon)
        for i, shape in enumerate(result_shape[:-1]):
            reshape_1_shapes.append(shape)
            reshape_1_indices.append(i)

        # Add to the lists the chunk dimensions.
        # For example, for lat: (e.g., clat, clon).
        # For the first reshape we have the shapes e.g., (nalat, nalon, clon, ctime)
        # and the corresponding indices e.g. (0, 1, 1, 2)
        reshape_1_shapes.extend(list(chunk_sizes_to_index))
        reshape_1_indices.extend(list(chunk_indices))

        # Do first reshape
        result = result.reshape(reshape_1_shapes)

        # Transpose to bring together the same dims e.g., (nalat, nalon, clon, ctime) if they're not adjacent already
        # For lon, we'd go from (nalat, nalon, clat, ctime) -> (nalat, clat, nalon, ctime)
        # Get the indices that would sort reshape_1_indices
        sort_indices = np.argsort(reshape_1_indices)
        new_indices = np.sort(reshape_1_indices)  # Sort this to use later

        # Apply sort_indices to result.transpose()
        result = result.transpose(tuple(sort_indices))

        # Second reshape operation to multiply the dim that appears more than once
        # E.g., lat: (nalat, nalon * clon, ctime), lon: (nalat * clat, nalon, ctime)
        # First, find duplicates in reshape_1_indices
        seen_indices = []  # set()
        reshape_2_shapes = []
        for idx, value in enumerate(new_indices):
            if value not in seen_indices:
                # If unique value
                reshape_2_shapes.append(result.shape[idx])
                seen_indices.append(value)
            else:
                # If duplicate value, multiply the corresponding element in
                # result.shape with idx that's the same
                reshape_2_shapes[np.where(seen_indices == value)[0][0]] *= result.shape[
                    idx
                ]
        # Apply the second reshape to result
        result = result.reshape(reshape_2_shapes)

        # Compute cumsum over the accumulation dim (dim_idx), skip for batch_dim
        if dim_i != batch_dim_idx:
            for i in dim_idx:
                result = result.cumsum(axis=i)

        # Final transpose to user-specified order, except for batch dim (e.g., time)
        if dim_i != batch_dim_idx:
            result = result.transpose(
                data_info_dict["accumulation_dim_orders_idx"][dim_i]
            )

        # Save result to Zarr
        assigning_slice = []
        for i, dim_order_idx in enumerate(
            data_info_dict["accumulation_dim_orders_idx"][dim_i]
        ):
            if dim_order_idx == batch_dim_idx:
                if dim_i != batch_dim_idx:
                    assigning_slice.append(slice(batch_idx_start, batch_idx_end, None))
                else:
                    assigning_slice.append(slice(idx_acc, idx_acc + 1, None))
                    # Expand a dimension for result at this idx
                    result = np.expand_dims(result, axis=i)
            else:
                assigning_slice.append(slice(None, None, None))

        if weights_flag:
            dim_name = data_info_dict["accumulation_weight_names"][dim_i]
        else:
            dim_name = data_info_dict["accumulation_names"][dim_i]
        if dim_i == batch_dim_idx:
            dim_name += "_temp"
        data_info_dict["accumulation_group"][dim_name][tuple(assigning_slice)] = result

        # Update current_idx in flattened outputs
        current_idx = end_idx

    return current_idx


def compute_assemble_zarr(
    data_info_dict,
    variable_array_dask,
    batch_dim_idx,
    batch_idx_start,
    batch_idx_end,
):
    """
    Compute and assemble Zarr arrays based on the computed block sums.

    This function computes and assembles Zarr arrays based on the computed block sums for a specified batch
    dimension range. It extracts data and weights from the block sums and updates corresponding Zarr arrays.

    Args:
        data_info_dict (dict): A dictionary containing various information about the data and accumulations.
        variable_array_dask (dask.array.Array): A Dask array representing the variable data.
        batch_dim_idx (int): Index of the batch dimension.
        batch_idx_start (int): Start index for the batch dimension slice.
        batch_idx_end (int): End index for the batch dimension slice.

    """

    idx_acc = int(
        batch_idx_start / data_info_dict["variable_array_chunks"][batch_dim_idx]
    )

    # Compute block sums using compute_block_sum function
    slice_list = [slice(None)] * variable_array_dask.ndim
    slice_list[batch_dim_idx] = slice(batch_idx_start, batch_idx_end)
    variable_block = variable_array_dask[tuple(slice_list)]

    block_sums = variable_block.map_blocks(
        compute_functions.compute_block_sum,
        data_info_dict["fill_value"],
        data_info_dict["accumulation_dimensions"],
        chunks=data_info_dict["variable_array_chunks"],
        dimension_arrays_dict=data_info_dict["dimension_arrays_dict"],
    ).compute()

    # Extract data and weights from block sums and update Zarr arrays
    current_idx = extract_data(
        data_info_dict,
        block_sums,
        batch_dim_idx,
        batch_idx_start,
        batch_idx_end,
        idx_acc,
        current_idx=0,
        weights_flag=False,
    )

    current_idx = extract_data(
        data_info_dict,
        block_sums,
        batch_dim_idx,
        batch_idx_start,
        batch_idx_end,
        idx_acc,
        current_idx=current_idx,
        weights_flag=True,
    )
    return


def setup_batch_dimension(
    data_info_dict,
    batch_dim_idx,
    batch_dim_idx_2,
    n_threads,
):
    """
    Setup and configure Zarr arrays for batch dimension processing.

    This function sets up and configures Zarr arrays for processing the batch dimension of the data. It creates
    new datasets and arrays, adapts shapes and chunks for the batch dimension, and prepares data structures for
    parallel processing.

    Args:
        data_info_dict (dict): A dictionary containing various information about the data and accumulations.
        batch_dim_idx (int): Index of the batch dimension.
        batch_dim_idx_2 (int): Index of the secondary batch dimension to batch over.
        n_threads (int): Number of threads for parallel processing.

    Returns:
        dict: A dictionary containing batch-related configuration and datasets.

    """

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    accumulation_group = data_info_dict["accumulation_group"]
    data_type = data_info_dict["data_type"]
    shape = data_info_dict["shape"]

    acc_dim_name = data_info_dict["accumulation_names"][batch_dim_idx]
    acc_dim_name_temp = acc_dim_name + "_temp"
    acc_dim_name_weight = data_info_dict["accumulation_weight_names"][batch_dim_idx]
    acc_dim_name_temp_weight = acc_dim_name_weight + "_temp"

    attributes_dim = accumulation_group[acc_dim_name].attrs["_ARRAY_DIMENSIONS"]
    attributes_stride = accumulation_group[acc_dim_name].attrs["_ACCUMULATION_STRIDE"]

    # Always a 1-element tuple e.g., (2,)
    batch_dim_stride = data_info_dict["accumulation_strides"][batch_dim_idx][0]

    # Create final dataset to run compute for the batch dimension (e.g., time)
    num_chunks_final = int(
        data_info_dict["num_chunks"][batch_dim_idx] / batch_dim_stride
    )
    num_batches = int(
        data_info_dict["variable_array_chunks"][batch_dim_idx_2] / n_threads
    )
    batch_size_per_thread = int(shape[batch_dim_idx_2] / n_threads)

    # Make final shape and chunks for this dim
    new_batch_array_shape = []
    new_batch_array_chunks = []
    # e.g., (2,)
    dim_idx_tuple = data_info_dict["accumulation_dimensions"][batch_dim_idx]
    # e.g., # (0, 2, 1)
    dim_order_idx = data_info_dict["accumulation_dim_orders_idx"][batch_dim_idx]
    for idx_val in dim_order_idx:
        if idx_val in dim_idx_tuple:
            # Append number of chunks in the accumulation dimension
            new_batch_array_shape.append(num_chunks_final)
            new_batch_array_chunks.append(
                int(data_info_dict["variable_array_chunks"][idx_val])
            )
        elif idx_val == batch_dim_idx_2:
            new_batch_array_shape.append(shape[idx_val])
            new_batch_array_chunks.append(num_batches)
        else:
            new_batch_array_shape.append(shape[idx_val])
            new_batch_array_chunks.append(shape[idx_val])

    batch_dataset = accumulation_group.create_dataset(
        acc_dim_name,
        shape=tuple(new_batch_array_shape),
        chunks=tuple(new_batch_array_chunks),
        compressor=compressor,
        dtype=data_type,
        # Filter goes here after figuring out how to handle it
        # filters=[AccumulationDeltaFilter(accumulation_dimensions[batch_dim_idx], accumulation_strides[batch_dim_idx], accumulation_dim_orders_idx[batch_dim_idx])],
        overwrite=True,
    )
    # Copy attribute file to new dataset
    batch_dataset.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
    batch_dataset.attrs["_ACCUMULATION_STRIDE"] = attributes_stride

    # Do the same for weights
    batch_dataset_weight = accumulation_group.create_dataset(
        acc_dim_name_weight,
        shape=tuple(new_batch_array_shape),
        chunks=tuple(new_batch_array_chunks),
        compressor=compressor,
        # Filter goes here after figuring out how to handle it
        # filters=[AccumulationDeltaFilter(accumulation_dimensions[batch_dim_idx], accumulation_strides[batch_dim_idx], accumulation_dim_orders_idx[batch_dim_idx])],
        dtype=data_type,
        overwrite=True,
    )
    batch_dataset_weight.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
    batch_dataset_weight.attrs["_ACCUMULATION_STRIDE"] = attributes_stride

    # Make dask data and weight arrays
    batch_array_dask = da.from_array(
        accumulation_group[acc_dim_name_temp],
        accumulation_group[acc_dim_name_temp].shape,
    )

    batch_array_dask_weight = da.from_array(
        accumulation_group[acc_dim_name_temp_weight],
        accumulation_group[acc_dim_name_temp_weight].shape,
    )

    batch_dict = {
        "dim_order_idx": dim_order_idx,
        "batch_dim_stride": batch_dim_stride,
        "new_batch_array_chunks": new_batch_array_chunks,
        "batch_size_per_thread": batch_size_per_thread,
        "batch_dataset": batch_dataset,
        "batch_dataset_weight": batch_dataset_weight,
        "batch_array_dask": batch_array_dask,
        "batch_array_dask_weight": batch_array_dask_weight,
    }
    return batch_dict
