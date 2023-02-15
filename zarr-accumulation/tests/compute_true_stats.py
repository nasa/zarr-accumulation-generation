import sys
import zarr
import numpy as np
import dask.array as da

sys.path.append("../data_preparation/")
from codec_filter import (
    DeltaLat,
    DeltaLon,
    DeltaTime,
)

if __name__ == "__main__":
    store_path = "../data_preparation/data/GPM_3IMERGHH_06_precipitationCal_out"
    true_store = zarr.DirectoryStore(store_path)
    z_true = zarr.open(true_store, mode="r")

    dimension_stats_dict = {
        "lat": {},
        "latlon": {},
        "latlonw": {},
        "lon": {},
        "time": {},
        "timew": {},
        # "latw": {},
        # "lonw": {},
    }

    for key, val in dimension_stats_dict.items():
        print(f"\nDimension: {key}")
        selected_z_true = z_true[key]

        da_true = da.from_array(selected_z_true)
        shape_true = selected_z_true.shape
        chunks_true = selected_z_true.chunks
        min_true = da_true.min().compute()
        max_true = da_true.max().compute()
        mean_true = da_true.mean().compute()
        std_true = da_true.std().compute()

        dimension_stats_dict[key]["shape"] = shape_true
        dimension_stats_dict[key]["chunks"] = chunks_true
        dimension_stats_dict[key]["min"] = min_true
        dimension_stats_dict[key]["max"] = max_true
        dimension_stats_dict[key]["mean"] = mean_true
        dimension_stats_dict[key]["std"] = std_true

    np.save("dimension_stats_dict.npy", dimension_stats_dict)
