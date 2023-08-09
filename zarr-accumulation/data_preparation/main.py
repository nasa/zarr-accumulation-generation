import os
import s3fs
import zarr
import numpy as np
import dask.array as da
from dask import compute
from numcodecs import Blosc
from multiprocessing import Process
from codec_filter import AccumulationDeltaFilter
from collections import OrderedDict
from itertools import combinations
import utilities.compute as compute_functions
import utilities.assemble as assemble_functions

compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
s3 = s3fs.S3FileSystem()


def create_test_data():
    # Create artificial data
    np.random.seed(0)
    clat, clon, ctime = 36, 72, 100
    chunks = (clat, clon, ctime)
    nlat, nlon, ntime = 144, 288, 200
    shape = (nlat, nlon, ntime)

    data_path = os.path.join(
        "data",
        "test_data",
    )
    root = zarr.open(data_path)
    variable_data = np.random.rand(nlat, nlon, ntime).astype("f4")
    variable_data[variable_data < 0.2] = -99
    root.create_dataset(
        "variable", shape=shape, chunks=chunks, data=variable_data, overwrite=True
    )
    root.create_dataset(
        "latitude",
        shape=nlat,
        chunks=clat,
        data=np.arange(-89.95, 90, 0.1)[:nlat],
        overwrite=True,
    )
    root.create_dataset(
        "longitude",
        shape=nlon,
        chunks=clon,
        data=np.arange(-179.95, 179.95, 0.1)[:nlon],
        overwrite=True,
    )
    time_data = np.arange(
        np.datetime64("2000-06-01"),
        np.datetime64("2000-12-26"),
        np.timedelta64(30, "m"),
        dtype="datetime64[m]",
    )[:ntime]
    root.create_dataset(
        "time", shape=ntime, chunks=ctime, data=time_data, overwrite=True
    )

    # Run helper.py script
    from subprocess import call

    call(
        [
            "python",
            f"../data_preparation/helper.py",
            "--path",
            "data/test_data",
        ]
    )
    return data_path


def construct_data_info_lists(
    variable_array,
    dim_combinations,
    acc_attrs,
    dim_to_idx,
    accumulation_group,
    batch_dim_idx,
):
    # Get the original data type
    data_type = variable_array.dtype.char + str(variable_array.dtype.itemsize)
    # print("data_type:", data_type)

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

    # print("accumulation_names:", accumulation_names)
    # print("accumulation_weight_names:", accumulation_weight_names)
    # print("accumulation_dimensions:", accumulation_dimensions)

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
    # print("accumulation_strides:", accumulation_strides)
    # print("accumulation_dim_orders:", accumulation_dim_orders)
    # print("accumulation_dim_orders_idx:", accumulation_dim_orders_idx, "\n")

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
        for idx_val in dim_order_idx:
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
    # print("array_shapes:", array_shapes)
    # print("array_chunks:", array_chunks, "\n")

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
    # Get dimension arrays to eventually pass into weight_computation_function()
    dimension_arrays = sorted(root.arrays())
    for array_i, array in enumerate(dimension_arrays):
        if array[0] == "variable":
            # Already got variable above, so remove from this list
            dimension_arrays.pop(array_i)
    dimension_arrays_dict = dict(dimension_arrays)
    # print("dimension_arrays_dict:", dimension_arrays_dict)

    # Get accumulation group
    accumulation_group = root["variable_accumulation_group"]

    # Assuming the attributes are pre-defined, read them in as a dictionary
    acc_attrs = OrderedDict(accumulation_group.attrs)
    dimensions = acc_attrs["_ACCUMULATION_GROUP"].keys()
    # print("acc_attrs:", acc_attrs, "\n")

    # Construct dictionaries as mappings between dimension name and index
    dim_to_idx = {}
    for dim_i, dim in enumerate(dimensions):
        dim_to_idx[dim] = dim_i
    # print("dim_to_idx:", dim_to_idx)

    # Swap key and value
    idx_to_dim = {v: k for k, v in dim_to_idx.items()}
    # print("idx_to_dim:", idx_to_dim, "\n")

    # Contruct a combination list of the dimensions
    dim_combinations = []
    for i in range(1, len(dimensions) + 1):
        dim_combinations += list(combinations(dimensions, i))
    # print("dim_combinations:", dim_combinations, "\n")

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
    # print("strides:", strides)
    variable_array_chunks = strides * np.array(data_info_dict["chunks"])
    # print("variable_array_chunks", variable_array_chunks, "\n")
    # Get number of chunks
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

    return (data_info_dict, variable_array_dask)


def update_zarr_arrays(data_info_dict, batch_dim_idx):
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
        # print(arr_shape, arr_chunks)

        # For the batch dim (e.g., time and time weight), first create a temp dataset
        if dim_i == batch_dim_idx:
            dim += "_temp"
            dim_weight += "_temp"
            # print("temp dim", dim)
            # print("temp dim_weight", dim_weight)

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
                # Filter goes here after figuring out how to handle it
                overwrite=True,
            )
        else:
            dataset = data_info_dict["accumulation_group"].create_dataset(
                dim,
                shape=arr_shape,
                chunks=arr_chunks,
                compressor=compressor,
                dtype=data_info_dict["data_type"],
                # Filter goes here after figuring out how to handle it
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
            # Filter goes here after figuring out how to handle it
            # May not be needed for weights
            overwrite=True,
        )
        dataset_weight.attrs["_ARRAY_DIMENSIONS"] = attributes_dim
        dataset_weight.attrs["_ACCUMULATION_STRIDE"] = attributes_stride
    return


def compute_write_zarr(
    data_info_dict,
    variable_array_dask,
    batch_dim_idx,
    batch_idx_start,
    batch_idx_end,
):
    idx_acc = int(
        batch_idx_start / data_info_dict["variable_array_chunks"][batch_dim_idx]
    )
    # print("idx_acc:", idx_acc)

    # Compute
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
    # print("block_sums.shape:", block_sums.shape)

    # Extract data for data and weights
    current_idx = assemble_functions.extract_data(
        data_info_dict,
        block_sums,
        batch_dim_idx,
        batch_idx_start,
        batch_idx_end,
        idx_acc,
        current_idx=0,
        weights_flag=False,
    )
    # print("current_idx", current_idx)

    current_idx = assemble_functions.extract_data(
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


def multiprocessing_loop(
    start, end, batch_size, compute_function, *compute_function_args
):
    processes = []
    for i in range(start, end):
        batch_idx_start = i * batch_size
        batch_idx_end = (i + 1) * batch_size

        # Append idx's to args (which is a tuple so we need to get the first element only)
        args_tup = compute_function_args[0] + (
            batch_idx_start,
            batch_idx_end,
        )

        print(f"Range: {batch_idx_start} -- {batch_idx_end}")
        process = Process(
            target=compute_function,
            args=args_tup,
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def main():
    # Setting variables - hardcoded for now, should become variables the user sets
    batch_size = 100
    batch_dim_idx = 2  # np.argmax(shape) Turn this into the largest dimension, keeping time now to validate with old code
    batch_dim_idx_2 = 0
    n_threads = 9
    data_path = "data/GPM_3IMERGHH_06_precipitationCal"

    # Open input Zarr store. If test mode, create test data on the fly.
    test_mode = True
    if test_mode:
        data_path = create_test_data()
    store_input = zarr.DirectoryStore(data_path)
    root = zarr.open(store_input)
    variable_array = root["variable"]

    # Process data parameters and build a dictionary of data info
    data_info_dict, variable_array_dask = process_data_params(
        variable_array, root, batch_dim_idx
    )
    # print(data_info_dict.keys())

    # Update Zarr arrays' shape and chunks by making new dataset with copying attribute files over
    update_zarr_arrays(data_info_dict, batch_dim_idx)

    # Compute batch info
    batch_dim_chunk_size = data_info_dict["variable_array_chunks"][batch_dim_idx]
    batch_dim_num_chunks = int(
        data_info_dict["shape"][batch_dim_idx] / batch_dim_chunk_size
    )
    # print(
    #     "batch_dim_chunk_size:",
    #     batch_dim_chunk_size,
    #     "\nbatch_dim_num_chunks:",
    #     batch_dim_num_chunks,
    #     "\n",
    # )

    # Run entire dataset
    print("Processing dataset... ")
    for batch_start in range(0, batch_dim_num_chunks, batch_size):
        print(f"Batch: {batch_start}")
        batch_end = min(batch_start + batch_size, batch_dim_num_chunks)
        multiprocessing_loop(
            batch_start,
            batch_end,
            batch_dim_chunk_size,
            compute_write_zarr,
            (
                data_info_dict,
                variable_array_dask,
                batch_dim_idx,
            ),
        )

    # Final compute and assembly for the batch dimension - batch_dim_idx (e.g., time)
    batch_dict = assemble_functions.setup_batch_dimension(
        data_info_dict,
        batch_dim_idx,
        batch_dim_idx_2,
        n_threads,
    )

    print("\nProcessing batch dimension... ")
    multiprocessing_loop(
        0,
        n_threads,
        batch_dict["batch_size_per_thread"],
        compute_functions.compute_batch_dimension,
        (
            data_info_dict["data_type"],
            batch_dict,
            batch_dim_idx,
            batch_dim_idx_2,
        ),
    )


if __name__ == "__main__":
    main()
