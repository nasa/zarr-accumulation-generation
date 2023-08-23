import numpy as np
import dask.array as da
from numcodecs import Blosc
from itertools import combinations
from collections import OrderedDict
from codec_filter import AccumulationDeltaFilter


def construct_data_info_lists(
    variable_array,
    dim_combinations,
    acc_attrs,
    dim_to_idx,
    accumulation_group,
    batch_dim_idx,
):
    """
    Constructs various information lists and dictionaries about the input variable array
    and its associated accumulation attributes.

    Args:
        variable_array (numpy.ndarray): The input variable array.
        dim_combinations (list of tuples): List of tuples representing combinations of dimensions.
        acc_attrs (dict): Dictionary containing accumulation attributes.
        dim_to_idx (dict): Dictionary mapping dimension names to their indices.
        accumulation_group (h5py.Group): Zarr group containing accumulation arrays and metadata.
        batch_dim_idx (int): Index of the batch dimension.

    Returns:
        dict: A dictionary containing various information about the data and accumulation attributes.

    """

    # Get the original data type
    data_type = variable_array.dtype.char + str(variable_array.dtype.itemsize)

    # Get fill value
    fill_value = variable_array.fill_value

    # Get shape and chunks
    shape = variable_array.shape
    chunks = variable_array.chunks

    # Construct spec lists of accumulation array names and the indices
    accumulation_names = []
    accumulation_weight_names = []
    accumulation_dimensions = []
    for tup in dim_combinations:
        temp_dim = acc_attrs["_ACCUMULATION_GROUP"]
        dim_indices = []
        for dim in tup:
            temp_dim = temp_dim[dim]
            dim_indices.append(dim_to_idx[dim])

        if temp_dim:  # If dictionary is not empty for these dimensions
            accumulation_names.append(temp_dim["_DATA_WEIGHTED"])
            accumulation_weight_names.append(temp_dim["_WEIGHTS"])
            accumulation_dimensions.append(tuple(dim_indices))

    # Get stride info and create arrays for length/shape/chunks
    accumulation_strides = []
    accumulation_dim_orders = []
    accumulation_dim_orders_idx = []
    for dim in accumulation_names:
        stride = tuple(
            [
                s
                for s in dict(accumulation_group[dim].attrs)["_ACCUMULATION_STRIDE"]
                if s != 0
            ]
        )
        accumulation_strides.append(stride)

        dim_order = []
        dim_order_idx = []
        for o in dict(accumulation_group[dim].attrs)["_ARRAY_DIMENSIONS"]:
            dim_order.append(o)
            dim_order_idx.append(dim_to_idx[o])
        accumulation_dim_orders.append(tuple(dim_order))
        accumulation_dim_orders_idx.append(tuple(dim_order_idx))

    # Adapting these lists to the dim order the user has in dataset attribute files
    array_shapes = []
    array_chunks = []
    for dim_i, dim_idx_tuple in enumerate(accumulation_dimensions):
        stride = accumulation_strides[dim_i]
        dim_order_idx = accumulation_dim_orders_idx[dim_i]

        array_shape = []
        array_chunk = []
        count = 0
        for idx_val in dim_order_idx:
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

    # Put data and accumulation attributes in a dictionary
    data_info_dict = {
        "data_type": data_type,
        "fill_value": fill_value,
        "shape": shape,
        "chunks": chunks,
        "accumulation_names": accumulation_names,
        "accumulation_weight_names": accumulation_weight_names,
        "accumulation_dimensions": accumulation_dimensions,
        "accumulation_strides": accumulation_strides,
        "accumulation_dim_orders": accumulation_dim_orders,
        "accumulation_dim_orders_idx": accumulation_dim_orders_idx,
        "array_shapes": array_shapes,
        "array_chunks": array_chunks,
    }
    return data_info_dict


def process_data_params(variable_array, root, batch_dim_idx):
    """
    Process data and accumulation parameters to prepare for computation.

    This function processes various data parameters and constructs dictionaries and arrays
    necessary for subsequent computations.

    Args:
        variable_array (numpy.ndarray): The input variable array.
        root (Zarr.Group): The root Zarr group containing relevant attributes.
        batch_dim_idx (int): Index of the batch dimension.

    Returns:
        tuple: A tuple containing two elements:
            - data_info_dict (dict): A dictionary containing various information about the data and accumulations.
            - variable_array_dask (dask.array.core.Array): A dask array containing the input variable data.

    """

    # Get dimension arrays to eventually pass into weight_computation_function()
    dimension_arrays = sorted(root.arrays())
    for array_i, array in enumerate(dimension_arrays):
        if array[0] == "variable":
            # Already got variable above, so remove from this list
            dimension_arrays.pop(array_i)
    dimension_arrays_dict = dict(dimension_arrays)

    # Get accumulation group
    accumulation_group = root["variable_accumulation_group"]

    # Assuming the attributes are pre-defined, read them in as a dictionary
    acc_attrs = OrderedDict(accumulation_group.attrs)
    dimensions = acc_attrs["_ACCUMULATION_GROUP"].keys()

    # Construct dictionaries as mappings between dimension name and index
    dim_to_idx = {}
    for dim_i, dim in enumerate(dimensions):
        dim_to_idx[dim] = dim_i

    # Swap key and value
    idx_to_dim = {v: k for k, v in dim_to_idx.items()}

    # Contruct a combination list of the dimensions
    dim_combinations = []
    for i in range(1, len(dimensions) + 1):
        dim_combinations += list(combinations(dimensions, i))

    # Get dict of data info
    data_info_dict = construct_data_info_lists(
        variable_array,
        dim_combinations,
        acc_attrs,
        dim_to_idx,
        accumulation_group,
        batch_dim_idx,
    )

    # Update chunks after applying strides
    strides = []
    for tup_i, tup in enumerate(data_info_dict["accumulation_strides"]):
        # Batch dimension (e.g., time) - apply stride at assembly step, so stride=1 here
        if tup_i == batch_dim_idx:
            strides.append(1)
        else:
            strides.append(tup[0])
    strides = tuple(np.array(strides)[: len(data_info_dict["chunks"])])
    variable_array_chunks = strides * np.array(data_info_dict["chunks"])
    num_chunks = tuple(np.array(data_info_dict["shape"]) / variable_array_chunks)

    # Convert data to dask array
    variable_array_dask = da.from_array(
        variable_array.astype("f8"), variable_array_chunks
    )

    # Append more info to dict
    data_info_dict["accumulation_group"] = accumulation_group
    data_info_dict["idx_to_dim"] = idx_to_dim
    data_info_dict["dimension_arrays_dict"] = dimension_arrays_dict
    data_info_dict["variable_array_chunks"] = variable_array_chunks
    data_info_dict["num_chunks"] = num_chunks

    return data_info_dict, variable_array_dask


def update_zarr_arrays(data_info_dict, batch_dim_idx):
    """
    Update Zarr arrays' shape and chunks by creating new datasets and copying attribute files to prepare for
    accumulation computation.

    This function iterates through the dimensions and their weight names in the provided data_info_dict,
    and creates new Zarr datasets with updated shape and chunks based on the provided information.
    It also copies over the necessary attribute files to the new datasets.

    Args:
        data_info_dict (dict): A dictionary containing various information about the data and accumulations.
        batch_dim_idx (int): Index of the batch dimension.

    """

    # Define Blosc compressor settings
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    # Update Zarr arrays' shape and chunks by making new dataset with copying attribute files over
    for dim_i, (dim, dim_weight) in enumerate(
        zip(
            data_info_dict["accumulation_names"],
            data_info_dict["accumulation_weight_names"],
        )
    ):
        # Creating new dataset with copying the attributes over
        attributes_dim = data_info_dict["accumulation_group"][dim].attrs[
            "_ARRAY_DIMENSIONS"
        ]
        attributes_stride = data_info_dict["accumulation_group"][dim].attrs[
            "_ACCUMULATION_STRIDE"
        ]

        arr_shape = data_info_dict["array_shapes"][dim_i]
        arr_chunks = data_info_dict["array_chunks"][dim_i]

        # For the batch dim (e.g., time and time weight), first create a temp dataset
        if dim_i == batch_dim_idx:
            dim += "_temp"
            dim_weight += "_temp"

        if (
            dim_i == batch_dim_idx
            or len(data_info_dict["accumulation_dimensions"][dim_i]) > 1
        ):
            dataset = data_info_dict["accumulation_group"].create_dataset(
                dim,
                shape=arr_shape,
                chunks=arr_chunks,
                compressor=compressor,
                dtype=data_info_dict["data_type"],
                overwrite=True,
            )
        else:
            dataset = data_info_dict["accumulation_group"].create_dataset(
                dim,
                shape=arr_shape,
                chunks=arr_chunks,
                compressor=compressor,
                dtype=data_info_dict["data_type"],
                filters=[
                    AccumulationDeltaFilter(
                        data_info_dict["accumulation_dimensions"][dim_i],
                        data_info_dict["accumulation_strides"][dim_i],
                        data_info_dict["accumulation_dim_orders_idx"][dim_i],
                    )
                ],
                overwrite=True,
            )
        dataset.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
        dataset.attrs["_ACCUMULATION_STRIDE"] = attributes_stride

        dataset_weight = data_info_dict["accumulation_group"].create_dataset(
            dim_weight,
            shape=arr_shape,
            chunks=arr_chunks,
            compressor=compressor,
            dtype=data_info_dict["data_type"],
            overwrite=True,
        )
        dataset_weight.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
        dataset_weight.attrs["_ACCUMULATION_STRIDE"] = attributes_stride
