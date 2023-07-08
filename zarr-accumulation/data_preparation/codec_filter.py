import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray
import numpy as np


class AccumulationDeltaFilter(Codec):
    codec_id = 'accumulation_delta_filter'

    def __init__(self, accumulation_dimension, accumulation_stride, accumulation_dim_order_idx):
        '''
        accumulation_dimensions: [(0,), (1,), (2,), (0, 1)]
        accumulation_strides: [(2,), (2,), (2,), (2, 2)]
        accumulation_dim_orders: [('latitude', 'time', 'longitude'), ('longitude', 'time', 'latitude'), ('latitude', 'time', 'longitude'), ('latitude', 'longitude', 'time')]
        accumulation_dim_orders_idx: [(0, 2, 1), (1, 2, 0), (0, 2, 1), (0, 1, 2)] 
        ZHL  0 acc_lat
        [25, 2000, 3600] [25, 200, 72]
        acc_lat [25, 2000, 3600] [25, 200, 72]
        ZHL  1 acc_lon
        [25, 2000, 1800] [25, 200, 36]
        acc_lon [25, 2000, 1800] [25, 200, 36]
        ZHL  2 acc_time
        [1800, 10, 3600] [36, 1, 72]
        acc_time_temp [1800, 10, 3600] [36, 1, 72]
        ZHL  3 acc_lat_lon
        [25, 25, 2000] [25, 25, 200]
        acc_lat_lon [25, 25, 2000] [25, 25, 200]
        variable_array_chunks [ 72 144 200] 
        '''
        #print("ZHL: ",accumulation_dimension, accumulation_stride, accumulation_dim_order_idx)
        self.accumulation_dimension = accumulation_dimension
        self.accumulation_stride = accumulation_stride
        self.accumulation_dim_order_idx = accumulation_dim_order_idx
        return None


    def encode(self, buf):
        accumulation_dimension_idx = [index for index, item in enumerate(self.accumulation_dim_order_idx) if item in self.accumulation_dimension]
        accumulation_chunk_shape = list(buf.shape)
        for idx in accumulation_dimension_idx:
            accumulation_chunk_shape[idx] = 1
        accumulation_chunk_shape = tuple(accumulation_chunk_shape)
        o = np.zeros(accumulation_chunk_shape, dtype=buf.dtype)
        slices = [slice(None)] * len(buf.shape)
        for idx in accumulation_dimension_idx:
            slices[idx] = slice(None, -1)
        try:
            # IMPORTANT: ONLY 1-D filter suppoted now
            buf1 = np.concatenate((o, buf[tuple(slices)]), axis=accumulation_dimension_idx[0])
            out = buf-buf1
        except:
            print(buf.shape, accumulation_chunk_shape, slices, accumulation_dimension_idx)
            raise
        return out

    def decode(self, buf, out=None):
        accumulation_dimension_idx = [index for index, item in enumerate(self.accumulation_dim_order_idx) if item in self.accumulation_dimension]
        enc = np.frombuffer(buf, dtype=buf.dtype).reshape(buf.shape)
        # IMPORTANT: ONLY 1-D filter suppoted now
        out = np.cumsum(enc, axis=accumulation_dimension_idx[0], out=out)
        return out

class DeltaLat(Codec):
    codec_id = 'delta_of_lat'

    def encode(self, buf):
        (_, a, b) = buf.shape
        o = np.zeros((1, a, b), dtype='f4')
        buf1 = np.concatenate((o, buf[:-1, :, :]), axis=0)
        out = buf-buf1
        return out

    def decode(self, buf, out=None):
        enc = np.frombuffer(buf, dtype='f4').reshape((25, 200, 144))
        out = np.cumsum(enc, axis=0, out=out)
        return out

class DeltaLon(Codec):
    codec_id = 'delta_of_lon'

    def encode(self, buf):
        (_, a, b) = buf.shape
        o = np.zeros((1, a, b), dtype='uint8')
        buf1 = np.concatenate((o, buf[:-1, :, :]), axis=0)
        '''
        (a, _, b) = buf.shape
        o = np.zeros((a, 1, b), dtype='uint8')
        buf1 = np.concatenate((o, buf[:, :-1, :]), axis=1)
        '''
        out = buf-buf1
        return out

    def decode(self, buf, out=None):
        enc = np.frombuffer(buf, dtype='f4').reshape((25, 200, 72))
        out = np.cumsum(enc, axis=0, out=out)
        return out

class DeltaTime(Codec):
    codec_id = 'delta_of_delta'

    def encode(self, buf):
        (a, _, b) = buf.shape
        o = np.zeros((a, 1, b), dtype=buf.dtype)
        buf1 = np.concatenate((o, buf[:, :-1, :]), axis=1)
        out = buf-buf1
        return out

    def decode(self, buf, out=None):
        enc = np.frombuffer(buf, dtype='f4').reshape((8, 200, 3600))
        out = np.cumsum(enc, axis=1, out=out)
        return out

class DeltaTimeW(Codec):
    codec_id = 'delta_of_delta_w'

    def encode(self, buf):
        (a, _, b) = buf.shape
        o = np.zeros((a, 1, b), dtype=buf.dtype)
        buf1 = np.concatenate((o, buf[:, :-1, :]), axis=1)
        out = buf-buf1
        return out

    def decode(self, buf, out=None):
        enc = np.frombuffer(buf, dtype='uint32').reshape((8, 200, 3600))
        out = np.cumsum(enc, axis=1, out=out, dtype='uint32')
        return out

numcodecs.register_codec(AccumulationDeltaFilter)
numcodecs.register_codec(DeltaLat)
numcodecs.register_codec(DeltaLon)
numcodecs.register_codec(DeltaTime)
numcodecs.register_codec(DeltaTimeW)
