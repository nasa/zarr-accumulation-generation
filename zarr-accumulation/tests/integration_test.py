import os
import sys
import zarr
import sure
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

        # Ground-truth checksums 
        # true_checksums = {
        #     "lat": "54c86fe1b52cba22942f6336ec3ba90bbd0d1e93", 
        #     "latlon": "52698f884f334ab88366d014f61621d2aa83516f", 
        #     "latlonw": "e570ba799faef0011cf3259ee3738ece5b0d9f31", 
        #     "lon": "fff15a819256f0f22cfb7554ba6aac4fbd9979ef", 
        #     "time": "806a2117442c3b5a22ab15fa452caa46975f1a7f", 
        #     "timew": "6b66aa41ff769fd470136022889e2edfcc3afcef", 
        #     "latw": "a68fee36129b5b96f4e5c534a77c01c312f02198", 
        #     "lonw": "8cbc91a53692acc9f3ac7a4beedc93faf6003429"
        # }

        true_checksums = {"lat": "09bd285e1b7973448e0cbe5212c61a5b0ff3f20e", "latlon": "eb362183ca3ccffebc2e12c1aac8caf565b14b48", "latlonw": "61bb5ec126b2010b827ace00c40b8ef57ac35d02", "lon": "e61cfd72e2d922260d97a2aac52c0da36faad43b", "time": "72e8b494e22bd59d0e7b1551c14a8bcdc2cb343f", "timew": "58e34c97ffa66dbddc3ad7a9b87a1e988298ea72", "latw": "386571ad13162abb00432a0c65dffb94819e0f1c", "lonw": "8dd79674382290f9e111782aa20439ae1a8eae09"}

        # Validate
        output_path = os.path.join(
            os.getcwd(), "data", "test_out",
        )
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
