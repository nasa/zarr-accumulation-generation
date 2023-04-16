import time
import math
import base64
import s3fs
from codec_filter import DeltaLat, DeltaLon, DeltaTime
import copy
import sys
import zarr
import dask
from dask import compute
import dask.array as da
from multiprocessing import Process
from threading import Thread
from threading import Thread
from numcodecs import Blosc, Delta
import numpy as np

compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
s3 = s3fs.S3FileSystem()


def compute_block_sum(block, block_info=None, wd=None, axis=0):
    if not block_info:
        return block
    (s0, s1, s2) = block.shape
    (i1, i2) = block_info[0]["array-location"][0]
    mask = block >= 0
    w = mask * (wd[i1:i2].reshape(s0, 1, 1))
    ow = block * w

    olat_sm = ow.sum(axis=0)
    olon_sm = ow.sum(axis=1)
    olatlon_sm = olat_sm.sum(axis=0)
    olatlon_wt = w.sum(axis=(0, 1))
    otime_sm = ow.sum(axis=2)
    otime_wt = w.sum(axis=2)

    output = np.concatenate(
        (
            olat_sm.flatten(),
            olon_sm.flatten(),
            olatlon_sm.flatten(),
            olatlon_wt.flatten(),
            otime_sm.flatten(),
            otime_wt.flatten(),
        )
    )
    output = output.reshape(1, 1, len(output))
    return output


def f_latlon_ptime(
    dd,
    zlat,
    zlatw,
    zlon,
    zlonw,
    zlatlon,
    zlatlonw,
    ztimetemp,
    ztimetempw,
    nlat,
    nlon,
    ntime,
    clat,
    clon,
    ctime,
    nalat,
    nalon,
    wd,
    a,
    b,
):

    # some locals
    idx_acc_time = int(a / ctime)

    nalat = int(nlat / clat)
    nalon = int(nlon / clon)
    num_chunks = nalat * nalon

    # compute
    t0 = time.time()
    data = (
        dd[:, :, a:b]
        .map_blocks(compute_block_sum, wd=wd, chunks=(clat, clon, ctime))
        .compute()
    )
    print("compute used: ", time.time() - t0)
    t0 = time.time()

    # extract data
    idx_0 = clon * ctime
    olat = (
        data[:, :, :idx_0]
        .reshape((nalat, nlon, -1))
        .cumsum(axis=0)
        .transpose((0, 2, 1))
        .astype("float32")
    )

    idx_1 = idx_0 + (clat * ctime)
    olon = (
        data[:, :, idx_0:idx_1]
        .transpose((1, 0, 2))
        .reshape((nalon, nlat, -1))
        .cumsum(axis=0)
        .transpose((0, 2, 1))
        .astype("float32")
    )

    idx_2 = idx_1 + ctime
    olatlon = data[:, :, idx_1:idx_2].cumsum(axis=0).cumsum(axis=1).astype("float32")

    idx_3 = idx_2 + ctime
    olatlonw = data[:, :, idx_2:idx_3].cumsum(axis=0).cumsum(axis=1).astype("float32")

    idx_4 = idx_3 + (clat * clon)
    otimetemp = data[:, :, idx_3:idx_4].reshape((nalat, nalon, clat, clon)).transpose((0,2,1,3)).reshape((nlat, nlon, 1))

    idx_5 = idx_4 + (nlat * nlon)
    otimetempw = data[:, :, idx_4:idx_5].reshape((nalat, nalon, clat, clon)).transpose((0,2,1,3)).reshape((nlat, nlon, 1))

    # save to zarr
    zlat[:, a:b, :] = olat
    zlon[:, a:b, :] = olon
    zlatlon[:, :, a:b] = olatlon
    zlatlonw[:, :, a:b] = olatlonw
    ztimetemp[:, :, idx_acc_time : idx_acc_time + 1] = otimetemp
    ztimetempw[:, :, idx_acc_time : idx_acc_time + 1] = otimetempw

    print("save used: ", time.time() - t0)
    return


def f_time(dz, dzw, ztime, ztimew, calat_in_time, natime, catime, nlon, a, b):
    ztime[a:b, :, :] = (
        dz[a:b, :, :natime]
        .cumsum(axis=2)[:, :, 1::2]
        .transpose((0, 2, 1))
        .rechunk((calat_in_time, catime, nlon))
        .astype("f4")
        .compute()
    )
    ztimew[a:b, :, :] = (
        dzw[a:b, :, :natime]
        .cumsum(axis=2)[:, :, 1::2]
        .transpose((0, 2, 1))
        .rechunk((calat_in_time, catime, nlon))
        .astype("uint32")
        .compute()
    )
    return


if __name__ == "__main__":
    start = time.time()

    test_mode = True
    if test_mode:
        from codec_filter_small import DeltaLat, DeltaLon, DeltaTime

        # Locals
        store_output = zarr.DirectoryStore("data/test_out")

        # Generate random data
        np.random.seed(0)
        chunks = (36, 72, 100)
        z = np.random.rand(144, 288, 200)
        z[z < 0.2] = -99
        (nlat, nlon, ntime) = z.shape
        (clat, clon, ctime) = chunks
    else:
        # Locals
        store_input = zarr.DirectoryStore("data/GPM_3IMERGHH_06_precipitationCal")
        store_output = zarr.DirectoryStore("data/GPM_3IMERGHH_06_precipitationCal_out")

        # Read zarr
        z = zarr.open(store_input, mode="r")["variable"]
        shape = z.shape
        chunks = z.chunks
        (nlat, nlon, ntime) = shape
        (clat, clon, ctime) = chunks

    root = zarr.open(store_output, mode="w")
    root_local = root

    clat *= 2  # NOTE coarse
    clon *= 2  # NOTE coarse
    dd = da.from_array(z.astype("f8"), (clat, clon, ctime))

    # HACK
    ntime = ntime
    print("nlat/nlon/ntime: ", nlat, nlon, ntime)
    print("clat/clon/ctime: ", clat, clon, ctime)

    # Weight array (1D)
    weight = np.cos(np.deg2rad([np.arange(-89.95, 90, 0.1)]))[:, :nlat].reshape(
        nlat, 1, 1
    )
    wd = da.from_array(weight)
    # dd *= wd
    print("read zarr took: ", time.time() - start)
    start = time.time()

    # Create time acc zarr
    natime = int(ntime / ctime)
    nalat = int(nlat / clat)
    nalon = int(nlon / clon)
    print("nalat/nalon/natime", nalat, nalon, natime)
    zlat = (
        root.create_dataset(
            "lat",
            shape=(nalat, ntime, nlon),
            chunks=(nalat, ctime, clon),
            compressor=compressor,
            filters=[DeltaLat()],
            dtype="f4",
        )
        if "lat" not in root
        else root.lat
    )
    zlatw = (
        root.create_dataset(
            "latw",
            shape=(nalat, ntime, nlon),
            chunks=(nalat, ctime, clon),
            compressor=compressor,
            dtype="|S9",
        )
        if "latw" not in root
        else root.latw
    )
    zlon = (
        root.create_dataset(
            "lon",
            shape=(nalon, ntime, nlat),
            chunks=(nalon, ctime, clat),
            compressor=compressor,
            filters=[DeltaLon()],
            dtype="f4",
        )
        if "lon" not in root
        else root.lon
    )
    zlonw = (
        root.create_dataset(
            "lonw",
            shape=(nalon, ntime, nlat),
            chunks=(nalon, ctime, clat),
            compressor=compressor,
            dtype="|S18",
        )
        if "lonw" not in root
        else root.lonw
    )
    zlatlon = (
        root.create_dataset(
            "latlon",
            shape=(nalat, nalon, ntime),
            chunks=(nalat, nalon, ctime),
            compressor=compressor,
            dtype="f4",
        )
        if "latlon" not in root
        else root.latlon
    )
    zlatlonw = (
        root.create_dataset(
            "latlonw",
            shape=(nalat, nalon, ntime),
            chunks=(nalat, nalon, ctime),
            compressor=compressor,
            dtype="f4",
        )
        if "latlonw" not in root
        else root.latlonw
    )
    ztimetemp = (
        root_local.create_dataset(
            "time_temp",
            shape=(nlat, nlon, natime),
            chunks=(clat, nlon, 1),
            compressor=compressor,
            dtype="f8",
        )
        if "time_temp" not in root_local
        else root_local.time_temp
    )
    ztimetempw = (
        root_local.create_dataset(
            "time_tempw",
            shape=(nlat, nlon, natime),
            chunks=(clat, nlon, 1),
            compressor=compressor,
            dtype="uint8",
        )
        if "time_tempw" not in root_local
        else root_local.time_tempw
    )

    # Compute
    # i=0; f_latlon_ptime(dd, zlat, zlatw, zlon, zlonw, zlatlon, zlatlonw, ztimetemp, ztimetempw, nlat, nlon, ntime, clat, clon, ctime, nalat, nalon, wd, i*ctime, (i+1)*ctime,); exit(0)
    batch_size = 100
    (natime_start, natime_end) = (0, natime)
    # (natime_start, natime_end) = (int(330000/200), natime)
    for batch_start in range(natime_start, natime_end, batch_size):
        print("Batch: ", batch_start)
        pp = []
        batch_end = min(batch_start + batch_size, natime_end)
        for i in range(batch_start, batch_end):
            print("Range: ", i * ctime, (i + 1) * ctime)
            p = Process(
                target=f_latlon_ptime,
                args=(
                    dd,
                    zlat,
                    zlatw,
                    zlon,
                    zlonw,
                    zlatlon,
                    zlatlonw,
                    ztimetemp,
                    ztimetempw,
                    nlat,
                    nlon,
                    ntime,
                    clat,
                    clon,
                    ctime,
                    nalat,
                    nalon,
                    wd,
                    i * ctime,
                    (i + 1) * ctime,
                ),
            )
            # p = Thread(target=f_latlon_ptime, args=(dd, zlat, zlatw, zlon, zlonw, zlatlon, zlatlonw, ztimetemp, ztimetempw, nlat, nlon, ntime, clat, clon, ctime, nalat, nalon, wd, i*ctime, (i+1)*ctime,))
            p.start()
            pp.append(p)
        for p in pp:
            p.join()
        print("compute lat/lon (partial time) took: ", time.time() - start)
        start = time.time()

    # Assemble time acc
    natime_final = int(natime / 2)
    catime = ctime
    calat_in_time = int(clat / 9)
    Nthreads = 9  # NOTE HARDCODED
    nlat_per_thread = int(nlat / Nthreads)
    ztime = (
        root.create_dataset(
            "time",
            shape=(nlat, natime_final, nlon),
            chunks=(calat_in_time, catime, nlon),
            compressor=compressor,
            filters=[DeltaTime()],
            dtype="f4",
        )
        if "time" not in root
        else root.time
    )
    ztimew = (
        root.create_dataset(
            "timew",
            shape=(nlat, natime_final, nlon),
            chunks=(calat_in_time, catime, nlon),
            compressor=compressor,
            filters=[DeltaTime()],
            dtype="uint32",
        )
        if "timew" not in root
        else root.timew
    )
    dz = da.from_array(ztimetemp, (nlat, nlon, natime))
    dzw = da.from_array(ztimetempw, (nlat, nlon, natime))
    pp = []
    for i in range(Nthreads):
        p = Process(
            target=f_time,
            args=(
                dz,
                dzw,
                ztime,
                ztimew,
                calat_in_time,
                natime,
                catime,
                nlon,
                i * nlat_per_thread,
                (i + 1) * nlat_per_thread,
            ),
        )
        p.start()
        pp.append(p)
    for p in pp:
        p.join()
    print("assemble time took: ", time.time() - start)
