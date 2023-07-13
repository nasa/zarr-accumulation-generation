import os
import sys
import zarr
import shutil
import runpy
import unittest

sys.path.append("../data_preparation/")
from codec_filter_small import (
    DeltaLat,
    DeltaLon,
    DeltaTime,
)


class Test_zarr_accumulation_entrypoint(unittest.TestCase):
    def test_random_data(self):
        # Call zarr accumulation entrypoint - still write out data to file
        script_path = os.path.join(
            os.getcwd(), "..", "data_preparation", "zarr_dask.py",
        )
        runpy.run_path(script_path, run_name="__main__")

        true_checksums = {
            "acc_lat": "58997e7630dcc122252bd8d31224a10e6b1e4480",
            "acc_lat_lon": "6892e3d8922a5915e9871de85f1d210f5a0bff4d",
            "acc_lon": "e6d6833b8f053b728b041db3dd1b4f12a8760b9e",
            "acc_time": "29a8c4d053e9d2ac1ca16397a315c30a7791e004",
            "acc_wt_lat": "2cf8b3a7d3b76ca785821b57da5fa7d2efbff82c",
            "acc_wt_lat_lon": "23d58a5a68141d074dfb8ee9e05be952d76ba05e",
            "acc_wt_lon": "09715b6db9cc3c00027f53f030e86e3bb8bc3162",
            "acc_wt_time": "7db7ff5d251a4e384608cc7e83b2a1679c648682",
        }

        # Validate
        output_path = os.path.join("data", "test_data", "variable_accumulation_group",)
        z_output = zarr.open(output_path, mode="r")

        for dim, checksum in true_checksums.items():
            print(f"Validating test output in dimension: {dim}")
            output_dim = z_output[dim]
            output_checksum = output_dim.hexdigest()
            assert output_checksum == checksum

        # Clean up local output store
        shutil.rmtree("data/test_data")


if __name__ == "__main__":
    unittest.main()
