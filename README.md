# Zarr-based chunk-level cumulative sums in reduced dimensions

This software is currently in beta pre-release.

## Overview <br>
Standard data analysis operations on multidimensional data in Zarr format often involve averaging along one or more dimensions. Standard methods requires a full scan of the data, making them computationally intensive, especially for large datasets. This algorithm provides an efficient and cost-effective approach for performing multidimensional averaging in Zarr format. It's particularly well-suited and useful for cloud or distributed systems but is also adaptable for local use. This method eliminates the need to read all data values by pre-computing cumulative sums ("accumulation" for short) along reduced data dimensions. The resulting accumulation data is saved as a small, adjustable set of auxiliary data on top of the untouched raw data and is used to quickly find the data averages along one or more dimensions. In this repository, we provide the source code for the generation of the accumulation data. The source code for using the accumulation data in averaging services is provided in a separate repository, [zarr-accumulation-service](https://github.com/nasa/zarr-accumulation-service). 


## Requirements <br>
The required Python packages for using the algorithm are provided in `requirements/core.txt` and can be installed using pip with the command: `pip install -r requirements/core.txt`. Additional packages required for running tests are provided in `requirements/core.txt`.


## How to use <br>

This code takes as input a Zarr store of multidimensional data stored under `zarr-accumulation/data_preparation/data/`. For example, a Zarr store "GPM_3IMERGHH_06_precipitationCal" of three dimensions (latitude, longitude, and time), is organized as follows:
```
.
└── zarr-accumulation
   └── data_preparation
      └── data
         └── GPM_3IMERGHH_06_precipitationCal
            ├── .zgroup
            ├── latitude
            │    ├── .zarray
            │    └── chunks
            ├── longitude
            │    ├── .zarray
            │    └── chunks
            ├── time
            │    ├── .zarray
            │    └── chunks
            └── variable
                ├── .zarray
                └── chunks
```

### 1. Set up Zarr store for accumulation generation with helper script
#### Input
`--path` or `-p` (str): Relative path of the Zarr store. Example: `data/GPM_3IMERGHH_06_precipitationCal/`.

#### Usage
The script `zarr-accumulation/data_preparation/helper.py` allows the user to customize and create an accumulation group inside the Zarr store to house the generated accumulation data. Inside the group, accumulation datasets are created with attribute files as part of the metadata. Actual chunks are produced in step 2. 

The Zarr attribute file of the accumulation group and of the accumulation data arrays are described in detail in the Zarr Enhancement Proposal, or [ZEP 5, Zarr-based Chunk-level Accumulation in Reduced Dimensions](https://github.com/zarr-developers/zeps/blob/main/draft/ZEP0005.md). 

Run the helper script with the following command: `python helper.py --path [Path of Zarr store]`. Example: `python helper.py --path data/GPM_3IMERGHH_06_precipitationCal/`.

#### Output 
An accumulation group will be created inside the Zarr store and the group includes accumulation datasets (e.g., `acc_lat`) and their metadata (i.e., `.zarray` and `.zattrs` files). The datasets will be populated with accumulation data chunks in the next step. The example Zarr store with accumulation along latitude, longitude, time, and latitude-longitude will have the following structure: 
```
.
└── zarr-accumulation
   └── data_preparation
      └── data
         └── GPM_3IMERGHH_06_precipitationCal
            ├── .zgroup
            ├── latitude
            │    ├── .zarray
            │    └── chunks
            ├── longitude
            │    ├── .zarray
            │    └── chunks
            ├── time
            │    ├── .zarray
            │    └── chunks
            ├── variable
            │    ├── .zarray
            │    └── chunks
            └── variable_accumulation_group
                ├── .zattrs
                ├── .zgroup
                ├── acc_lat
                │    ├── .zarray
                │    └── .zattrs
                ├── acc_lat_lon
                │    ├── .zarray
                │    └── .zattrs
                ├── acc_lon
                │    ├── .zarray
                │    └── .zattrs
                └── acc_time
                    ├── .zarray
                    └── .zattrs
```

### 2. Generate accumulation data
#### Input
- `--batch_size` (int): Batch size along the specified dimension (e.g., time). Default: 100.
- `--batch_dim_idx` (int): The index of the first batch dimension (e.g., `2` for time). Default: 2. 
- `--batch_dim_idx_2` (int): The index of the second batch dimension (e.g., `0` for latitude). Default: 0. 
- `--n_threads` (int): Number of threads. Default: 9. 
- `--data_path` (str): Relative path of the Zarr store. Example: `data/GPM_3IMERGHH_06_precipitationCal/`.

#### Usage 
The entrypoint script `zarr-accumulation/data_preparation/main.py` will take the above command-line arguments, processes data parameters, performs batch processing using parallel computation, to generate the accumulation data and write these data arrays to the accumulation group datasets. Run the script with the following command with user-specified arguments or defaults: `python main.py --batch_size [batch size] --batch_dim_idx [first index] --batch_dim_idx_2 [second index] --n_threads [number of threads] --data_path [path]`.

#### Output 
The datasets inside the accumulation group will be populated with the accumulation chunk data. For example, the `acc_lat` dataset will have the following chunks stored next to the metadata:
```
.
└── zarr-accumulation
   └── data_preparation
      └── data
         └── GPM_3IMERGHH_06_precipitationCal
            ├── ...
            └── variable_accumulation_group
                ├── .zattrs
                ├── .zgroup
                ├── acc_lat
                │    ├── .zarray
                │    ├── .zattrs
                │    ├── 0.0.0
                │    ├── 0.0.1
                │    ├── ...
                │    └── 0.9.49
                └── ...
```

## Testing <br>
The integration test using an auto-generated small 3-D dataset can be run from the home directory of this repository with the following command: `bin/test`. 

### Reference
Zhang, H., Hegde, M., Smit, C., Pham, L., Pagan, B., & Nguyen, D.M. (2023). [ZEP 5 — Zarr-based Chunk-level Accumulation in Reduced Dimensions](https://github.com/zarr-developers/zeps/blob/main/draft/ZEP0005.md). In <i>Zarr Enhancement Proposals Github Repository.</i>

Zhang, H. (2022, July). Zarr-based Chunk-level Accumulation in Reduced Dimensions. Presentation at <i>Earth Science Information Partners (ESIP) 2022 Summer Meeting</i>.

Zhang, H., Hegde, M., Smit, C., & Pham, L. (2021, December). Zarr-based Analysis-ready Data in the Cloud with Chunk-level Cumulative Sums. In <i>Advancing Earth and Space Science (AGU) Fall Meeting Abstracts</i> (Vol. 2021, pp. IN35D-0417).

