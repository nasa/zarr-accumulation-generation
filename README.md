# zarr-accumulation
This is the source code for the algorithm "Zarr-based chunk-level cumulative sums in reduced dimensions."

Common data analysis operations on multidimensional data in Zarr format typically includes averaging along one or more dimensions. Standard methods requires a full scan of the data and can be computationally expensive with large datasets. This algorithm provides a fast and cost-efficient method to perform multidimensional averaging services in Zarr on the cloud. This method eliminates the need to read all data values by pre-computing cumulative sums (accumulation) and weights along data dimensions at the chunk level. This accumulation data is saved as a small adjustable set of
auxiliary data on top of the untouched raw data and is used to quickly find the data averages along one or more dimensions.

## Testing
To run integration tests, navigate to `zarr-accumulation/zarr-accumulation/tests/` and run the command: `python integration_test.py`.