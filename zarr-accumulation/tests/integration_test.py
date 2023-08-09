import os
import sys
import zarr
import shutil
import unittest
import numpy as np
from subprocess import call

sys.path.append("../data_preparation/")
from codec_filter import AccumulationDeltaFilter


class Test_zarr_accumulation_entrypoint(unittest.TestCase):
    """
    Test class for validating the Zarr accumulation data prepration code.

    This class defines unit tests to validate the functionality of the Zarr accumulation data prepration code. It sets up
    random Zarr test data, runs the main entrypoint script, and validates the output Zarr accumulation arrays against expected
    checksums.

    """

    def setUp(self):
        """
        Set up random test data and environment for testing.

        This method creates random test data using Zarr, including variable data, latitude, longitude, and time
        information. It also runs the 'helper.py' script to prepare the data for accumulation computation.

        Returns:
            str: The path to the created data.

        """
        # Create random Zarr store
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

        call(
            [
                "python",
                f"../data_preparation/helper.py",
                "--path",
                "../tests/data/test_data",
            ]
        )
        return data_path

    def test_random_data(self):
        """
        Test the Zarr accumulation code with random data and validate the output.

        This method calls the Zarr accumulation entrypoint script, validates the output Zarr arrays against expected
        checksums, and cleans up the local test output store.

        """
        call(
            [
                "python",
                "../data_preparation/main.py",
                "--data_path",
                "data/test_data",
            ]
        )

        true_checksums = {
            "acc_lat": "89705ef22dd704d2161fa280a0a2e898f2013935",
            "acc_lat_lon": "6892e3d8922a5915e9871de85f1d210f5a0bff4d",
            "acc_lon": "820326a7a9148f3517dc282ab29187214f16674e",
            "acc_time": "29a8c4d053e9d2ac1ca16397a315c30a7791e004",
            "acc_wt_lat": "2cf8b3a7d3b76ca785821b57da5fa7d2efbff82c",
            "acc_wt_lat_lon": "23d58a5a68141d074dfb8ee9e05be952d76ba05e",
            "acc_wt_lon": "09715b6db9cc3c00027f53f030e86e3bb8bc3162",
            "acc_wt_time": "7db7ff5d251a4e384608cc7e83b2a1679c648682",
        }

        # Validate checksums of test output vs true checksums
        output_path = os.path.join(
            "data",
            "test_data",
            "variable_accumulation_group",
        )
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
