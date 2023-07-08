import numcodecs
from numcodecs.abc import Codec
import numpy as np


class AccumulationDeltaFilter(Codec):
    codec_id = 'accumulation_delta_filter'

    def __init__(self, accumulation_dimension, accumulation_stride, accumulation_dim_order_idx):
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

numcodecs.register_codec(AccumulationDeltaFilter)