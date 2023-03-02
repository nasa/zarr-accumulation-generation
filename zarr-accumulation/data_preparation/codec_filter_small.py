import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray
import numpy as np


class DeltaLat(Codec):
    codec_id = "delta_of_lat"

    def encode(self, buf):
        (_, a, b) = buf.shape
        o = np.zeros((1, a, b), dtype="f4")
        buf1 = np.concatenate((o, buf[:-1, :, :]), axis=0)
        out = buf - buf1
        return out

    def decode(self, buf, out=None):
        enc = np.frombuffer(buf, dtype="f4").reshape((2, 100, 144))
        out = np.cumsum(enc, axis=0, out=out)
        return out


class DeltaLon(Codec):
    codec_id = "delta_of_lon"

    def encode(self, buf):
        (_, a, b) = buf.shape
        o = np.zeros((1, a, b), dtype="uint8")
        buf1 = np.concatenate((o, buf[:-1, :, :]), axis=0)
        """
        (a, _, b) = buf.shape
        o = np.zeros((a, 1, b), dtype='uint8')
        buf1 = np.concatenate((o, buf[:, :-1, :]), axis=1)
        """
        out = buf - buf1
        return out

    def decode(self, buf, out=None):
        enc = np.frombuffer(buf, dtype="f4").reshape((2, 100, 72))
        out = np.cumsum(enc, axis=0, out=out)
        return out


class DeltaTime(Codec):
    codec_id = "delta_of_delta"

    def encode(self, buf):
        (a, _, b) = buf.shape
        o = np.zeros((a, 1, b), dtype=buf.dtype)
        buf1 = np.concatenate((o, buf[:, :-1, :]), axis=1)
        out = buf - buf1
        return out

    def decode(self, buf, out=None):
        enc = np.frombuffer(buf, dtype="f4").reshape((8, 100, 288))
        out = np.cumsum(enc, axis=1, out=out)
        return out


class DeltaTimeW(Codec):
    codec_id = "delta_of_delta_w"

    def encode(self, buf):
        (a, _, b) = buf.shape
        o = np.zeros((a, 1, b), dtype=buf.dtype)
        buf1 = np.concatenate((o, buf[:, :-1, :]), axis=1)
        out = buf - buf1
        return out

    def decode(self, buf, out=None):
        enc = np.frombuffer(buf, dtype="f4").reshape((8, 100, 288))
        out = np.cumsum(enc, axis=1, out=out, dtype="uint32")
        return out


numcodecs.register_codec(DeltaLat)
numcodecs.register_codec(DeltaLon)
numcodecs.register_codec(DeltaTime)
numcodecs.register_codec(DeltaTimeW)
