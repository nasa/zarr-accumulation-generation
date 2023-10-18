import numpy as np


def weight_computation_function(
    block, block_info, fill_value, dimension_arrays_dict, weight_dimension="latitude"
):
    """
    Compute weights and weighted blocks for a given data block.

    This function computes weights and applies them to the data block based on the specified weight_dimension.
    The weights are computed using a cosine transformation of the dimension array's values.

    Args:
        block (numpy.ndarray): The data block to compute weights for.
        block_info (dict): Information about the data block's location and attributes.
        fill_value: Fill value for the data block.
        dimension_arrays_dict (dict): Dictionary containing dimension arrays.
        weight_dimension (str): The dimension to use for computing weights. Default is "latitude".

    Returns:
        tuple: A tuple containing two elements:
            - weights (numpy.ndarray): Computed weights for the data block.
            - block_weighted (numpy.ndarray): The data block after applying weights.

    """

    (s0, _, _) = block.shape
    (i1, i2) = block_info[0]["array-location"][0]

    # Get dimension array for weight computation
    dimension_array = dimension_arrays_dict[weight_dimension]
    block_dimension_array = dimension_array[i1:i2]

    # Compute weights using a cosine transformation
    block_weights = np.cos(np.deg2rad(block_dimension_array))

    # Apply weights to the data block and create a mask
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
    """
    Compute weighted sums over specified dimensions for a given data block.

    Args:
        block (numpy.ndarray): The data block to compute sums for.
        fill_value: Fill value for the data block.
        accumulation_dimensions (list): List of indices representing dimensions for accumulation.
        block_info (dict or None): Information about the data block's location and attributes.
        dimension_arrays_dict (dict or None): Dictionary containing dimension arrays.

    Returns:
        numpy.ndarray: An array containing the computed sums and weight sums over the specified dimensions.

    """

    if not block_info:
        return block

    # Get weights and weighted block
    weights, block_weighted = weight_computation_function(
        block, block_info, fill_value, dimension_arrays_dict
    )

    # Get data sums
    outputs = []
    for dim_idx in accumulation_dimensions:
        output = block_weighted.sum(axis=dim_idx)
        outputs.append(output.flatten())

    # Get weight sums
    for dim_idx in accumulation_dimensions:
        output_weight = weights.sum(axis=dim_idx)
        outputs.append(output_weight.flatten())

    # Concatenate sums and reshape
    outputs = np.concatenate(outputs)
    outputs = outputs.reshape(len(outputs))
    for i in range(len(block.shape) - 1):
        outputs = np.expand_dims(outputs, axis=i)

    return outputs


def compute_batch_dimension(
    data_type,
    batch_dict,
    batch_dim_idx,
    batch_dim_idx_2,
    batch_idx_start,
    batch_idx_end,
):
    """
    Compute cumsum and update Zarr arrays along the batch dimension.

    This function computes cumulative sums along the batch dimension for both data and weights,
    and updates the corresponding Zarr arrays with the computed results.

    Args:
        data_type (str): The data type of the variable array for the computations.
        batch_dict (dict): A dictionary containing various batch-related information.
        batch_dim_idx (int): Index of the batch dimension.
        batch_dim_idx_2 (int): Index of second batch dimension.
        batch_idx_start (int): Start index for the batch dimension slice.
        batch_idx_end (int): End index for the batch dimension slice.

    """

    batch_array_dask = batch_dict["batch_array_dask"]
    batch_array_dask_weight = batch_dict["batch_array_dask_weight"]
    batch_dataset = batch_dict["batch_dataset"]
    batch_dataset_weight = batch_dict["batch_dataset_weight"]
    dim_order_idx = batch_dict["dim_order_idx"]
    batch_dim_stride = batch_dict["batch_dim_stride"]
    new_batch_array_chunks = batch_dict["new_batch_array_chunks"]

    # Create slices for indexing
    slice_list = [slice(None)] * batch_array_dask.ndim
    slice_list[batch_dim_idx_2] = slice(batch_idx_start, batch_idx_end)

    cumsum_axis = list(dim_order_idx).index(batch_dim_idx)
    slice_list_cumsum = [slice(None)] * batch_array_dask.ndim
    slice_list_cumsum[cumsum_axis] = slice(1, None, batch_dim_stride)

    # Perform cumsum and rechunk on data and update Zarr arrays
    result = (
        batch_array_dask[tuple(slice_list)]
        .cumsum(axis=cumsum_axis)[tuple(slice_list_cumsum)]
        .rechunk(new_batch_array_chunks)
        .astype(data_type)
        .compute()
    )
    batch_dataset[tuple(slice_list)] = result

    # Perform cumsum and rechunk on weights and update Zarr arrays
    result_weight = (
        batch_array_dask_weight[tuple(slice_list)]
        .cumsum(axis=cumsum_axis)[tuple(slice_list_cumsum)]
        .rechunk(new_batch_array_chunks)
        .astype(data_type)
        .compute()
    )
    batch_dataset_weight[tuple(slice_list)] = result_weight
    return
