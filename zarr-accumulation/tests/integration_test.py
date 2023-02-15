import sys
import zarr
import time
import sure
import shutil
import runpy
import subprocess
import unittest
import numpy as np
import dask.array as da

sys.path.append("../data_preparation/")
from codec_filter import (
    DeltaLat,
    DeltaLon,
    DeltaTime,
)


class Test_zarr_accumulation_entrypoint(unittest.TestCase):
    def test_GPM_3IMERGDF(self):
        start_time = time.time()

        # Call zarr accumulation entrypoint
        runpy.run_path("../data_preparation/zarr_dask.py", run_name="__main__")

        # Read in stats of ground-truth Zarr accumulation data
        true_stats = np.load("dimension_stats_dict.npy", allow_pickle="TRUE").item()

        # Validate test output against true statistics
        output_path = "data/GPM_3IMERGHH_06_precipitationCal_out"
        z_output = zarr.open(output_path, mode="r")

        for dim, stats in true_stats.items():
            print(f"Validating test output in dimension: {dim}")
            output_dim = z_output[dim]
            output_dim_da = da.from_array(output_dim)

            output_shape = output_dim.shape
            # Compare values using "sure" package - can also have some difference tolerance if desired
            stats["shape"].should.equal(output_shape)

            output_chunks = output_dim.chunks
            stats["chunks"].should.equal(output_chunks)

            output_min = output_dim_da.min().compute()
            stats["min"].should.equal(output_min)

            output_max = output_dim_da.max().compute()
            stats["max"].should.equal(output_max)

            output_mean = output_dim_da.mean().compute()
            stats["mean"].should.equal(output_mean)

            output_std = output_dim_da.std().compute()
            stats["std"].should.equal(output_std)

        # Clean up local output store - maybe this should be optional with flag
        shutil.rmtree(output_path)

        print(f"Integration test took: {time.time() - start_time}")


if __name__ == "__main__":
    unittest.main()
