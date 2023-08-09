import numpy as np


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
    batch_array_dask = batch_dict["batch_array_dask"]
    batch_array_dask_weight = batch_dict["batch_array_dask_weight"]
    batch_dataset = batch_dict["batch_dataset"]
    batch_dataset_weight = batch_dict["batch_dataset_weight"]
    dim_order_idx = batch_dict["dim_order_idx"]
    batch_dim_stride = batch_dict["batch_dim_stride"]
    new_batch_array_chunks = batch_dict["new_batch_array_chunks"]

    # Do cumsum for the batch dimension (e.g., time) and update zarr array
    slice_list = [slice(None)] * batch_array_dask.ndim
    slice_list[batch_dim_idx_2] = slice(batch_idx_start, batch_idx_end)

    cumsum_axis = list(dim_order_idx).index(batch_dim_idx)
    slice_list_cumsum = [slice(None)] * batch_array_dask.ndim
    slice_list_cumsum[cumsum_axis] = slice(1, None, batch_dim_stride)

    # Operate on data
    result = (
        batch_array_dask[tuple(slice_list)]
        .cumsum(axis=cumsum_axis)[tuple(slice_list_cumsum)]  # No transpose needed
        .rechunk(new_batch_array_chunks)
        .astype(data_type)
        .compute()
    )
    batch_dataset[tuple(slice_list)] = result

    # Operate on weights
    result_weight = (
        batch_array_dask_weight[tuple(slice_list)]
        .cumsum(axis=cumsum_axis)[tuple(slice_list_cumsum)]  # No transpose needed
        .rechunk(new_batch_array_chunks)
        .astype(data_type)
        .compute()
    )
    batch_dataset_weight[tuple(slice_list)] = result_weight
    return
