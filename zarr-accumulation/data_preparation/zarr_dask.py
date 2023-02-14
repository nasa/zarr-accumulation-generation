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

compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
s3 = s3fs.S3FileSystem()

def compute_block_sum(block, block_info=None, wd=None, axis=0):
    if not block_info:
        return block
    (s0,s1,s2) = block.shape
    (i1, i2) = block_info[0]["array-location"][0]
    mask = block>=0
    w = mask * (wd[i1:i2].reshape(s0,1,1))
    ow = block * w
    olat_sm = ow.sum(axis=0)
    olon_sm = np.concatenate( ( ow.sum(axis=1), np.zeros((s1-s0, s2), dtype=block.dtype) ), axis=0)
    olat_wt = np.frombuffer(bytes(bytearray(np.packbits(mask, axis=0).T)),dtype='|S9').reshape(s2,s1).T
    olon_wt = np.frombuffer(bytes(bytearray(np.packbits(mask, axis=1).transpose(2,0,1))),dtype='|S9').reshape(s2,s1).T
    '''
    olon_wt = np.concatenate(
                ( np.frombuffer(bytes(bytearray(np.packbits(mask, axis=1).transpose(2,0,1))),dtype='|S18').reshape(s2,s0).T,
                  np.zeros((s1-s0, s2), dtype='|S18') ),
                axis=0)
    '''
    olatlon_sm = olat_sm.sum(axis=0)
    olatlon_wt = w.sum(axis=(0,1))
    otime_sm = ow.sum(axis=2)
    otime_wt = w.sum(axis=2)
    otime_sm = np.resize(np.append(otime_sm.flatten(), olatlon_sm), s1*s2).reshape(s1,s2) 
    otime_wt = np.resize(np.append(otime_wt.flatten(), olatlon_wt), s1*s2).reshape(s1,s2)
    '''
    otime_sm = np.resize(np.append(ow.sum(axis=2).flatten(), olat_sm.sum(axis=0)), s1*s2).reshape(s1,s2) 
    otime_wt = np.resize(np.append(w.sum(axis=2).flatten(), w.sum(axis=(0,1))), s1*s2).reshape(s1,s2)
    '''
    
    o = np.core.records.fromarrays(
        [olat_sm, olat_wt, olon_sm, olon_wt, otime_sm, otime_wt],
        dtype=[("lat_sm", "f8"), ("lat_wt", "|S9"), ("lon_sm", "f8"), ("lon_wt", "|S9"), ("time_sm", "f8"), ("time_wt", "f8")],
    ).reshape(1, s1, s2)
    return o


def f_latlon_ptime(dd, zlat, zlatw, zlon, zlonw, zlatlon, zlatlonw, ztimetemp, ztimetempw, nlat, nlon, ntime, clat, clon, ctime, nalat, nalon, wd, a, b):

    # some locals
    idx_acc_time = int(a/ctime)

    # compute
    t0=time.time()
    data = dd[:, :, a:b].map_blocks(compute_block_sum, wd=wd, chunks=(clat, clon, ctime)).compute()
    print("compute used: ",time.time()-t0); t0=time.time()

    # extract data
    olon  = np.array([data['lon_sm'][(i%nalat), int(i/nalat)*clon : int(i/nalat)*clon +clat, :].flatten() for i in range(nalat*nalon)]).reshape(nalon, nlat, -1).transpose((0, 2, 1))
    #olonw = np.array([data['lon_wt'][(i%nalat), int(i/nalat)*clon : int(i/nalat)*clon +clat, :].flatten() for i in range(nalat*nalon)]).reshape(nalon, nlat, -1).transpose((0, 2, 1))
    olonw = np.array([np.frombuffer(data['lon_wt'][(i%nalat), int(i/nalat)*clon : int(i/nalat+1)*clon, :].tobytes(), dtype='|S18').flatten() for i in range(nalat*nalon)]).reshape(nalon, nlat, -1).transpose((0, 2, 1))
    otimetemp = np.concatenate([np.concatenate([data['time_sm'][ilat, (ilon*clon):((ilon+1)*clon)].flatten()[:clat*clon].reshape(clat,clon) for ilon in range(nalon)], axis=1) for ilat in range(nalat)], axis=0).reshape(nlat, nlon, 1)
    otimetempw = np.concatenate([np.concatenate([data['time_wt'][ilat, (ilon*clon):((ilon+1)*clon)].flatten()[:clat*clon].reshape(clat,clon) for ilon in range(nalon)], axis=1) for ilat in range(nalat)], axis=0).reshape(nlat, nlon, 1)
    olatlon = np.concatenate([[data['time_sm'][ilat, (ilon*clon):((ilon+1)*clon), :].flatten()[(clat*clon):(clat*clon+ctime)] for ilon in range(nalon)] for ilat in range(nalat)]).reshape(nalat, nalon, ctime)
    olatlonw = np.concatenate([[data['time_wt'][ilat, (ilon*clon):((ilon+1)*clon), :].flatten()[(clat*clon):(clat*clon+ctime)] for ilon in range(nalon)] for ilat in range(nalat)]).reshape(nalat, nalon, ctime)
    print("extract used: ",time.time()-t0); t0=time.time()

    # save to zarr
    zlat[:,a:b,:]  = data['lat_sm'][:, :, :].cumsum(axis=0).transpose((0,2,1)).astype("float32")
    zlatw[:,a:b,:] = data['lat_wt'][:].transpose((0,2,1))
    zlon[:,a:b,:]  = olon[:, :, :].cumsum(axis=0).astype("float32")
    zlonw[:,a:b,:] = olonw[:, :, :]
    zlatlon[:,:,a:b] =  olatlon[:, :, :(b-a)].cumsum(axis=0).cumsum(axis=1).astype("float32")
    zlatlonw[:,:,a:b] =  olatlonw[:, :, :(b-a)].cumsum(axis=0).cumsum(axis=1).astype("float32")
    ztimetemp[:,:,idx_acc_time:idx_acc_time+1] = otimetemp[:, :, :]
    ztimetempw[:,:,idx_acc_time:idx_acc_time+1] = otimetempw[:, :, :]
    print("save used: ",time.time()-t0)
    return


def f_time(dz, dzw, ztime, ztimew, calat_in_time, natime, catime, nlon, a, b):
    ztime[a:b,:,:] = dz[a:b, :, :natime].cumsum(axis=2)[:,:,1::2].transpose((0,2,1)).rechunk((calat_in_time, catime, nlon)).astype("f4").compute()
    ztimew[a:b,:,:] = dzw[a:b, :, :natime].cumsum(axis=2)[:,:,1::2].transpose((0,2,1)).rechunk((calat_in_time, catime, nlon)).astype("uint32").compute()
    return

if __name__ == '__main__':

    # Locals
    flag_remote = False
    if flag_remote:
        store_input = s3fs.S3Map(root='s3://uat-giovanni-cache/zarr/GPM_3IMERGHH_06_precipitationCal/',
                s3=s3fs.S3FileSystem(),
                check=False)
    else:
        store_input = zarr.DirectoryStore('data/GPM_3IMERGHH_06_precipitationCal')
    store_output = zarr.DirectoryStore('data/GPM_3IMERGHH_06_precipitationCal_out')
    root = zarr.open(store_output, mode='a')
    root_local = root

    # Read zarr
    start = time.time()
    z = zarr.open(store_input, mode='r')['variable']
    shape = z.shape
    chunks = z.chunks
    (nlat, nlon, ntime) = shape
    (clat, clon, ctime) = chunks
    clat *= 2 # NOTE coarse
    clon *= 2 # NOTE coarse
    dd = da.from_array(z.astype('f8'), (clat, clon, ctime))

    # HACK
    ntime = 1000
    print('nlat/nlon/ntime: ',nlat,nlon,ntime)
    print('clat/clon/ctime: ',clat,clon,ctime)

    # Weight array (1D)
    weight = np.cos(np.deg2rad([np.arange(-89.95,90,0.1)]))[:, :nlat].reshape(nlat,1,1)
    wd = da.from_array(weight)
    #dd *= wd
    print("read zarr took: ",time.time()-start); start = time.time()

    # Create time acc zarr
    natime = int(ntime/ctime)
    nalat = int(nlat/clat)
    nalon = int(nlon/clon)
    zlat = root.create_dataset('lat', shape=(nalat, ntime, nlon), chunks=(nalat, ctime, clon),
                                      compressor=compressor, filters=[DeltaLat()], dtype='f4') if 'lat' not in root else root.lat
    zlatw = root.create_dataset('latw', shape=(nalat, ntime, nlon), chunks=(nalat, ctime, clon),
                                        compressor=compressor, dtype='|S9') if 'latw' not in root else root.latw
    zlon = root.create_dataset('lon', shape=(nalon, ntime, nlat), chunks=(nalon, ctime, clat),
                                      compressor=compressor, filters=[DeltaLon()], dtype='f4') if 'lon' not in root else root.lon
    zlonw = root.create_dataset('lonw', shape=(nalon, ntime, nlat), chunks=(nalon, ctime, clat),
                                        compressor=compressor, dtype='|S18') if 'lonw' not in root else root.lonw
    zlatlon = root.create_dataset('latlon', shape=(nalat, nalon, ntime), chunks=(nalat, nalon, ctime),
                                            compressor=compressor, dtype='f4') if 'latlon' not in root else root.latlon
    zlatlonw = root.create_dataset('latlonw', shape=(nalat, nalon, ntime), chunks=(nalat, nalon, ctime),
                                              compressor=compressor, dtype='f4') if 'latlonw' not in root else root.latlonw
    ztimetemp = root_local.create_dataset('time_temp', shape=(nlat, nlon, natime), chunks=(clat, nlon, 1),
                                                 compressor=compressor, dtype='f8') if 'time_temp' not in root_local else root_local.time_temp
    ztimetempw = root_local.create_dataset('time_tempw', shape=(nlat, nlon, natime), chunks=(clat, nlon, 1),
                                                   compressor=compressor, dtype='uint8') if 'time_tempw' not in root_local else root_local.time_tempw

    # Compute
    #i=0; f_latlon_ptime(dd, zlat, zlatw, zlon, zlonw, zlatlon, zlatlonw, ztimetemp, ztimetempw, nlat, nlon, ntime, clat, clon, ctime, nalat, nalon, wd, i*ctime, (i+1)*ctime,); exit(0)
    batch_size = 100
    (natime_start, natime_end) = (0, natime)
    #(natime_start, natime_end) = (int(330000/200), natime)
    for batch_start in range(natime_start, natime_end, batch_size):
        print("Batch: ",batch_start)
        pp=[]
        batch_end = min(batch_start+batch_size, natime_end)
        for i in range(batch_start, batch_end):
            print("Range: ",i*ctime, (i+1)*ctime)
            p = Process(target=f_latlon_ptime, args=(dd, zlat, zlatw, zlon, zlonw, zlatlon, zlatlonw, ztimetemp, ztimetempw, nlat, nlon, ntime, clat, clon, ctime, nalat, nalon, wd, i*ctime, (i+1)*ctime,))
            #p = Thread(target=f_latlon_ptime, args=(dd, zlat, zlatw, zlon, zlonw, zlatlon, zlatlonw, ztimetemp, ztimetempw, nlat, nlon, ntime, clat, clon, ctime, nalat, nalon, wd, i*ctime, (i+1)*ctime,))
            p.start()
            pp.append(p)
        for p in pp:
            p.join()
        print("compute lat/lon (partial time) took: ",time.time()-start); start = time.time()

    # Assemble time acc
    natime_final = int(natime/2)
    catime = ctime
    calat_in_time = int(clat/9)
    Nthreads = 9 # NOTE HARDCODED
    nlat_per_thread = int(nlat/Nthreads)
    ztime = root.create_dataset('time', shape=(nlat, natime_final, nlon), chunks=(calat_in_time, catime, nlon),
                                        compressor=compressor, filters=[DeltaTime()], dtype='f4') if 'time' not in root else root.time
    ztimew = root.create_dataset('timew', shape=(nlat, natime_final, nlon), chunks=(calat_in_time, catime, nlon),
                                          compressor=compressor, filters=[DeltaTime()], dtype='uint32') if 'timew' not in root else root.timew
    dz = da.from_array(ztimetemp, (nlat, nlon, natime))
    dzw = da.from_array(ztimetempw, (nlat, nlon, natime))
    pp=[]
    for i in range(Nthreads):
        p = Process(target=f_time, args=(dz, dzw, ztime, ztimew, calat_in_time, natime, catime, nlon, i*nlat_per_thread, (i+1)*nlat_per_thread))
        p.start()
        pp.append(p)
    for p in pp:
        p.join()
    print("assemble time took: ",time.time()-start); start = time.time()

