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
            "lat": "09bd285e1b7973448e0cbe5212c61a5b0ff3f20e",
            "latw": "e05ee3aa6b10bb881754ca7ed1740de9cc55c49d",
            "lon": "e61cfd72e2d922260d97a2aac52c0da36faad43b",
            "lonw": "aa06ff043a5e5c578c44abe8f7188652e41c70a5",
            "latlon": "eb362183ca3ccffebc2e12c1aac8caf565b14b48",
            "latlonw": "61bb5ec126b2010b827ace00c40b8ef57ac35d02",
            "time": "72e8b494e22bd59d0e7b1551c14a8bcdc2cb343f",
            "timew": "58e34c97ffa66dbddc3ad7a9b87a1e988298ea72",
        }

        # Validate
        output_path = os.path.join(os.getcwd(), "data", "test_out",)
        z_output = zarr.open(output_path, mode="r")

        for dim, checksum in true_checksums.items():
            print(f"Validating test output in dimension: {dim}")
            output_dim = z_output[dim]
            output_checksum = output_dim.hexdigest()
            assert output_checksum == checksum

        # Clean up local output store
        shutil.rmtree(output_path)


if __name__ == "__main__":
    unittest.main()
