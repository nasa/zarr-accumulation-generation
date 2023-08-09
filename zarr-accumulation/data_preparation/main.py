import zarr
import argparse
from multiprocessing import Process
import utilities.compute as compute_functions
import utilities.assemble as assemble_functions
import utilities.setup as setup_functions


def setup_args():
    """
    Set up command line arguments for the data accumulation preparation script.

    This function defines and configures command line arguments using the `argparse` module. It provides descriptions,
    default values, and types for the command line arguments relevant to the data accumulation preparation process.

    Returns:
        argparse.Namespace: A namespace containing parsed command line arguments.

    """
    parser = argparse.ArgumentParser(description="Accumulation data preparation")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=100)
    parser.add_argument(
        "--batch_dim_idx", type=int, help="First batch dimension index", default=2
    )
    parser.add_argument(
        "--batch_dim_idx_2", type=int, help="Second batch dimension index", default=0
    )
    parser.add_argument("--n_threads", type=int, help="Number of threads", default=9)
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path of Zarr store",
        default="data/GPM_3IMERGHH_06_precipitationCal",
    )
    return parser.parse_args()


def multiprocessing_loop(
    start, end, batch_size, compute_function, *compute_function_args
):
    """
    Execute a computation function in parallel using multiple processes.

    This function creates and manages a pool of processes to execute a given computation function in parallel.
    It divides the computation into batches based on the provided range and batch size and distributes the work
    across multiple processes.

    Args:
        start (int): The starting index of the range of batches.
        end (int): The ending index of the range of batches (exclusive).
        batch_size (int): The size of each batch.
        compute_function (callable): The function to be executed in parallel.
        *compute_function_args: Variable-length arguments to be passed to the compute function.

    """
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


def main(args):
    """
    Main function to perform accumulation data preparation and write data arrays to the accumulation group.

    This function orchestrates the entire data preparation process. It takes command-line arguments,
    processes data parameters, updates accumulation Zarr arrays, and performs batch processing using parallel computation.

    Args:
        args (argparse.Namespace): Command-line arguments.

    """
    # Setting user-specified variables
    batch_size = args.batch_size
    batch_dim_idx = args.batch_dim_idx
    batch_dim_idx_2 = args.batch_dim_idx_2
    n_threads = args.n_threads
    data_path = args.data_path

    # Open Zarr store
    store_input = zarr.DirectoryStore(data_path)
    root = zarr.open(store_input)
    variable_array = root["variable"]

    # Process data parameters and build a dictionary of data info
    data_info_dict, variable_array_dask = setup_functions.process_data_params(
        variable_array, root, batch_dim_idx
    )

    # Update Zarr arrays' shape and chunks by making new dataset with copying attribute files over
    setup_functions.update_zarr_arrays(data_info_dict, batch_dim_idx)

    # Compute batch info
    batch_dim_chunk_size = data_info_dict["variable_array_chunks"][batch_dim_idx]
    batch_dim_num_chunks = int(
        data_info_dict["shape"][batch_dim_idx] / batch_dim_chunk_size
    )

    # Compute and assemble entire dataset looping over the batch dimension batch_dim_idx (e.g., time)
    print("Processing dataset... ")
    for batch_start in range(0, batch_dim_num_chunks, batch_size):
        print(f"Batch: {batch_start}")
        batch_end = min(batch_start + batch_size, batch_dim_num_chunks)
        multiprocessing_loop(
            batch_start,
            batch_end,
            batch_dim_chunk_size,
            assemble_functions.compute_assemble_zarr,
            (
                data_info_dict,
                variable_array_dask,
                batch_dim_idx,
            ),
        )

    # Compute and assembe the batch dimension looping over the second batch dimension batch_dim_idx_2 (e.g., latitude)
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
    args = setup_args()
    main(args)
