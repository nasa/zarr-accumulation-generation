import os
import sys
import zarr
import sure
import shutil
import runpy
import unittest
import numpy as np
from numpy import testing as numpy_testing

sys.path.append("../data_preparation/")
from codec_filter import (
    DeltaLat,
    DeltaLon,
    DeltaTime,
)


class Test_zarr_accumulation_entrypoint(unittest.TestCase):
    def test_GPM_3IMERGDF(self):
        # Call zarr accumulation entrypoint
        script_path = os.path.join(
            os.getcwd(), "..", "data_preparation", "zarr_dask.py",
        )
        runpy.run_path(script_path, run_name="__main__")

        # Stats of ground-truth Zarr accumulation data
        true_stats = {
            "lat": {"mean": 72.27080535888672, "std": 106.652587890625},
            "latlon": {"mean": 126743.4765625, "std": 127761.359375},
            "latlonw": {"mean": 1006204.1875, "std": 1006401.375},
            "lon": {"mean": 130.29505920410156, "std": 213.71470642089844},
            "time": {"mean": 45.3724479675293, "std": 95.65841674804688},
            "timew": {"mean": 347.1451639660494, "std": 267.3870121417042},
            "latw": {"mean": 2417489479736260.0, "std": 5.995023718241938e17},
            "lonw": {"mean": -628757442906794.6, "std": 4.65020815943922e17},
        }

        # Validate test output against true statistics
        output_path = os.path.join(
            os.getcwd(), "data", "GPM_3IMERGHH_06_precipitationCal_out",
        )
        z_output = zarr.open(output_path, mode="r")

        for dim, stats in true_stats.items():
            print(f"Validating test output in dimension: {dim}")
            output_dim = np.array(z_output[dim])
            if dim == "latw" or dim == "lonw":
                output_dim = np.frombuffer(output_dim, dtype=int)

            output_mean = output_dim.mean()
            numpy_testing.assert_allclose(stats["mean"], output_mean, rtol=1e-6)

            output_std = output_dim.std()
            numpy_testing.assert_allclose(stats["std"], output_std, rtol=1e-6)

        # Clean up local output store
        shutil.rmtree(output_path)


if __name__ == "__main__":
    unittest.main()
