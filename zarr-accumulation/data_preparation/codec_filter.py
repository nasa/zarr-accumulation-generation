import numcodecs
from numcodecs.abc import Codec
import numpy as np


class AccumulationDeltaFilter(Codec):
    """
    Implements the Accumulation Delta Filter codec.

    This codec performs accumulation delta filtering on multidimensional NumPy arrays.

    Attributes:
        accumulation_dimension (tuple): The dimensions along which accumulation delta filtering is performed.
        accumulation_stride (int): The stride of accumulation delta filtering.
        accumulation_dim_order_idx (tuple): The indices of the dimensions in the order they appear in the array.

    Methods:
        encode(buf): Encodes the input array using accumulation delta filtering.
        decode(buf, out=None): Decodes the input buffer using accumulation delta filtering.

    Note:
        Only 1-D filter is supported at the moment.
    """

    codec_id = "accumulation_delta_filter"

    def __init__(
        self, accumulation_dimension, accumulation_stride, accumulation_dim_order_idx
    ):
        """
        Initializes the AccumulationDeltaFilter codec.

        Args:
            accumulation_dimension (tuple): The dimensions along which accumulation delta filtering is performed.
            accumulation_stride (int): The stride of accumulation delta filtering.
            accumulation_dim_order_idx (tuple): The indices of the dimensions in the order they appear in the array.
        """
        self.accumulation_dimension = accumulation_dimension
        self.accumulation_stride = accumulation_stride
        self.accumulation_dim_order_idx = accumulation_dim_order_idx

    def encode(self, buf):
        """
        Encodes the input array using accumulation delta filtering.

        Args:
            buf (ndarray): The input array to be encoded.

        Returns:
            ndarray: The encoded array.
        """
        accumulation_dimension_idx = [
            index
            for index, item in enumerate(self.accumulation_dim_order_idx)
            if item in self.accumulation_dimension
        ]
        accumulation_chunk_shape = list(buf.shape)
        for idx in accumulation_dimension_idx:
            accumulation_chunk_shape[idx] = 1
        accumulation_chunk_shape = tuple(accumulation_chunk_shape)
        o = np.zeros(accumulation_chunk_shape, dtype=buf.dtype)
        slices = [slice(None)] * len(buf.shape)
        for idx in accumulation_dimension_idx:
            slices[idx] = slice(None, -1)
        try:
            # IMPORTANT: ONLY 1-D filter supported now
            buf1 = np.concatenate(
                (o, buf[tuple(slices)]), axis=accumulation_dimension_idx[0]
            )
            out = buf - buf1
        except:
            print(
                buf.shape, accumulation_chunk_shape, slices, accumulation_dimension_idx
            )
            raise
        return out

    def decode(self, buf, out=None):
        """
        Decodes the input buffer using accumulation delta filtering.

        Args:
            buf (ndarray): The input buffer to be decoded.
            out (ndarray, optional): Output array. If provided, the decoded array will be stored here.

        Returns:
            ndarray: The decoded array.
        """
        accumulation_dimension_idx = [
            index
            for index, item in enumerate(self.accumulation_dim_order_idx)
            if item in self.accumulation_dimension
        ]
        enc = np.frombuffer(buf, dtype=buf.dtype).reshape(buf.shape)
        # IMPORTANT: ONLY 1-D filter supported now
        out = np.cumsum(enc, axis=accumulation_dimension_idx[0], out=out)
        return out


# Register the AccumulationDeltaFilter codec
numcodecs.register_codec(AccumulationDeltaFilter)
