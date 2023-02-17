import os
import sys
import json
import zarr
import numpy as np

sys.path.append("../data_preparation/")
from codec_filter import (
    DeltaLat,
    DeltaLon,
    DeltaTime,
)

if __name__ == "__main__":
    store_path = os.path.join(
        os.getcwd(),
        "..",
        "data_preparation",
        "data",
        "GPM_3IMERGHH_06_precipitationCal_out",
    )

    true_store = zarr.DirectoryStore(store_path)
    z_true = zarr.open(true_store, mode="r")

    dimension_stats_dict = {
        "lat": {},
        "latlon": {},
        "latlonw": {},
        "lon": {},
        "time": {},
        "timew": {},
        "latw": {},
        "lonw": {},
    }

    for key, val in dimension_stats_dict.items():
        print(f"\nDimension: {key}")
        selected_z_true = np.array(z_true[key])
        if key == "latw" or key == "lonw":
            selected_z_true = np.frombuffer(selected_z_true, dtype=int)
        mean_true = selected_z_true.mean()
        std_true = selected_z_true.std()
        dimension_stats_dict[key]["mean"] = float(mean_true)
        dimension_stats_dict[key]["std"] = float(std_true)

    with open("dimension_stats_dict.json", "w") as outfile:
        json.dump(dimension_stats_dict, outfile)
