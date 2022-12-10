"""
Lazy frame wrapper. Efficient data structure to save each frame only once.
Can use LZ4 compression to optimizer memory usage.
"""
from typing import List, Tuple

import numpy as np


class LazyFrames(object):
    def __init__(self, frames: List, compress: bool = False):
        """
        :param frames: queue. ( Collection.dequeue)
        :param compress: compress frame
        """
        if compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.compress = compress

    def __array__(self) -> np.ndarray:
        """Makes the LazyFrames object convertible to a NumPy array
        """
        if self.compress:
            from lz4.block import decompress
            dtype = self._frames[0].dtype
            dshape = self._frames[0].shape
            frames = [np.frombuffer(decompress(_f), dtype=dtype).reshape(dshape) for _f in self._frames]
        else:
            frames = self._frames

        return np.stack(frames, axis=0)

    def __getitem__(self, index: int) -> np.ndarray:
        """Return frame at index
        """
        return self.__array__()[index]

    def __len__(self) -> int:
        """Return length of data structure
        """
        return len(self.__array__())

    def __eq__(self, other: np.ndarray) -> bool:
        """Compares if data structure is equivalent to another object
        """
        return self.__array__() == other

    @property
    def shape(self) -> Tuple:
        """Returns dimensions of other object
        """
        return self.__array__().shape

