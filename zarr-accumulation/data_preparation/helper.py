import zarr
import argparse
from collections import OrderedDict
from itertools import combinations

"""This helper script allows the user to create an accumulation group with an 
atribute file. Inside the group, accumulation datasets are created with attribute files. """

if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default="data/GPM_3IMERGHH_06_precipitationCal",
        help="Path of Zarr store",
    )
    args = parser.parse_args()
    path = args.path

    # Open input Zarr store
    store_input = zarr.DirectoryStore(path)
    root = zarr.open(store_input)
    variable_array = root["variable"]

    # Create accumulation group and the attribute file in JSON format
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

    # Assuming the attributes are pre-defined, read them in as a dictionary
    acc_attrs = OrderedDict(acc_group.attrs)
    dimensions = acc_attrs["_ACCUMULATION_GROUP"].keys()
    print("acc_attrs:", acc_attrs, "\n")

    # Construct dictionaries as mappings between dimension name and index
    dim_to_idx = {}
    for dim_i, dim in enumerate(dimensions):
        dim_to_idx[dim] = dim_i
    print("dim_to_idx:", dim_to_idx)

    # Contruct a combination list of the dimensions
    dim_combinations = []
    for i in range(1, len(dimensions) + 1):
        dim_combinations += list(combinations(dimensions, i))
    print("dim_combinations:", dim_combinations, "\n")

    # Construct spec lists of accumulation array names and the indices
    accumulation_names = []
    accumulation_weight_names = []
    for tup in dim_combinations:
        temp_dim = acc_attrs["_ACCUMULATION_GROUP"]
        dim_indices = []
        for dim in tup:
            temp_dim = temp_dim[dim]
            dim_indices.append(dim_to_idx[dim])

        if temp_dim:  # If dictionary is not empty for these dimensions
            accumulation_names.append(temp_dim["_DATA_WEIGHTED"])
            accumulation_weight_names.append(temp_dim["_WEIGHTS"])

    print("accumulation_names:", accumulation_names)
    print("accumulation_weight_names:", accumulation_weight_names)

    # Innitialize data arrays for accumulation datasets
    for dim in accumulation_names + accumulation_weight_names:
        dataset = acc_group.create_dataset(
            dim,
            shape=(0, 0, 0),  # Dummy empty shape, will update after getting stride info
            overwrite=True,
        )

        # Different dimensions order for each dataset
        if dim == "acc_lat" or dim == "acc_wt_lat":
            dimension_list = ["latitude", "time", "longitude"]
            stride_list = [2, 0, 0]
        if dim == "acc_lon" or dim == "acc_wt_lon":
            dimension_list = ["longitude", "time", "latitude"]
            stride_list = [2, 0, 0]
        if dim == "acc_lat_lon" or dim == "acc_wt_lat_lon":
            dimension_list = ["latitude", "longitude", "time"]
            stride_list = [2, 2, 0]
        if dim == "acc_time" or dim == "acc_wt_time":
            dimension_list = ["latitude", "time", "longitude"]
            stride_list = [0, 2, 0]
        dataset.attrs["_ARRAY_DIMENSIONS"] = dimension_list
        dataset.attrs["_ACCUMULATION_STRIDE"] = stride_list
