import os
import sys
import json
import zarr

sys.path.append("../data_preparation/")
from codec_filter_small import (
    DeltaLat,
    DeltaLon,
    DeltaTime,
)

if __name__ == "__main__":
    store_path = os.path.join(
        os.getcwd(), "..", "data_preparation", "data", "test_out",
    )

    true_store = zarr.DirectoryStore(store_path)
    z_true = zarr.open(true_store, mode="r")

    dimension_stats_dict = {
        "lat": None,
        "latw": None,
        "lon": None,
        "lonw": None,
        "latlon": None,
        "latlonw": None,
        "time": None,
        "timew": None,
    }

    for key, val in dimension_stats_dict.items():
        print(f"\nDimension: {key}")
        selected_z_true = z_true[key]
        dimension_stats_dict[key] = selected_z_true.hexdigest()

    with open("validation_checksums.json", "w") as outfile:
        json.dump(dimension_stats_dict, outfile)
