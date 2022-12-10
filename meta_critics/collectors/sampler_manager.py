from __future__ import annotations

import multiprocessing
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Generic, Optional, Tuple, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt

SharedMemoryLike = Union[str, SharedMemory]  # shared memory or name of shared memory
SharedT = TypeVar("SharedT", bound=np.generic)
SliceLike = Union[int, slice]
NDIndex = Tuple[int, ...]
NDSlice = Tuple[SliceLike, ...]  # does not guarantee at least one slice :-(
NDSliceLike = Union[SliceLike, NDSlice]


__all__ = ("SharedNDArray",)
__version__ = "1.0.0.post0"


class SharedNDArray(Generic[SharedT]):
    """Class to keep track of and retrieve the data in a shared array

    Attributes
    ----------
    shm
        SharedMemory object containing the data of the array
    shape
        Shape of the NumPy array
    dtype
        Type of the NumPy array. Anything that may be passed to the `dtype=` argument in `np.ndarray`.
    lock
        (Optional) multiprocessing.Lock to manage access to the SharedNDArray. This is only created if
        lock=True is passed to the constructor, otherwise it is set to `None`.

    A SharedNDArray object may be created either directly with a preallocated shared memory object plus the
    dtype and shape of the numpy array it represents:

    >>> from multiprocessing.shared_memory import SharedMemory
    >>> import numpy as np
    >>> from shared_ndarray2 import SharedNDArray
    >>> x = np.array([1, 2, 3])
    >>> shm = SharedMemory(name="x", create=True, size=x.nbytes)
    >>> arr = SharedNDArray(shm, x.shape, x.dtype)
    >>> arr[:] = x[:]  # copy x into the array
    >>> print(arr[:])
    [1 2 3]
    >>> shm.close()
    >>> shm.unlink()

    Or using a SharedMemoryManager either from an existing array or from arbitrary shape and nbytes:

    >>> from multiprocessing.managers import SharedMemoryManager
    >>> mem_mgr = SharedMemoryManager()
    >>> mem_mgr.start()  # Better yet, use SharedMemoryManager context manager
    >>> arr = SharedNDArray.from_shape(mem_mgr, x.shape, x.dtype)
    >>> arr[:] = x[:]  # copy x into the array
    >>> print(arr[:])
    [1 2 3]
    >>> # -or in one step-
    >>> arr = SharedNDArray.from_array(mem_mgr, x)
    >>> print(arr[:])
    [1 2 3]

    `SharedNDArray` does not subclass numpy.ndarray but rather generates an ndarray on-the-fly in get(),
    which is used in __getitem__ and __setitem__. Thus to access the data and/or use any ndarray methods
    get() or __getitem__ or __setitem__ must be used

    >>> arr.max()  # ERROR: SharedNDArray has no `max` method.
    Traceback (most recent call last):
        ....
    AttributeError: SharedNDArray object has no attribute 'max'. To access NumPy ndarray object use .get() method.
    >>> arr.get().max()  # (or arr[:].max())  OK: This gets an ndarray on which we can operate
    3
    >>> y = np.zeros(3)
    >>> y[:] = arr  # ERROR: Cannot broadcast-assign a SharedNDArray to ndarray `y`
    Traceback (most recent call last):
        ...
    ValueError: setting an array element with a sequence.
    >>> y[:] = arr[:]  # OK: This gets an ndarray that can be copied element-wise to `y`
    >>> mem_mgr.shutdown()
    """

    shm: SharedMemory
    shape: Tuple[int, ...]  # is a property
    dtype: np.dtype
    lock: Optional[multiprocessing.Lock]

    def __init__(
        self, shm: SharedMemoryLike, shape: Tuple[int, ...], dtype: npt.DTypeLike, *, lock: bool = False
    ):
        """Initialize a SharedNDArray object from existing shared memory, object shape, and dtype.

        To initialize a SharedNDArray object from a memory manager and data or shape, use the `from_array()
        or `from_shape()` classmethods.

        Parameters
        ----------
        shm
            `multiprocessing.shared_memory.SharedMemory` object or name for connecting to an existing block
            of shared memory (using SharedMemory constructor)
        shape
            Shape of the NumPy array to be represented in the shared memory
        dtype
            Data type for the NumPy array to be represented in shared memory. Any valid argument for
            `np.dtype` may be used as it will be converted to an actual `dtype` object.
        lock : bool, optional
            If True, create a multiprocessing.Lock object accessible with the `.lock` attribute, by default
            False.  If passing the `SharedNDArray` as an argument to a `multiprocessing.Pool` function this
            should not be used -- see this comment to a Stack Overflow question about `multiprocessing.Lock`:
            https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes#comment72803059_25558333

        Raises
        ------
        ValueError
            The SharedMemory size (number of bytes) does not match the product of the shape and dtype
            itemsize.
        """
        if isinstance(shm, str):
            shm = SharedMemory(name=shm, create=False)
        dtype = np.dtype(dtype)  # Try to convert to dtype
        if shm.size != dtype.itemsize * np.prod(shape):
            raise ValueError(
                "The SharedMemory object shm must have the same size as the product of the size of the dtype"
                " and the shape."
            )
        self.shm = shm
        self.dtype = dtype
        self._shape: Tuple[int, ...] = shape
        self.lock = multiprocessing.Lock() if lock else None

    def __repr__(self):
        # Like numpy's ndarray repr
        cls_name = self.__class__.__name__
        nspaces = len(cls_name) + 1
        array_repr = str(self.get())
        array_repr = array_repr.replace("\n", "\n" + " " * nspaces)
        return f"{cls_name}({array_repr}, dtype={self.dtype})"

    @classmethod
    def from_array(
        cls, mem_mgr: SharedMemoryManager, arr: npt.NDArray[SharedT], *, lock: bool = False
    ) -> SharedNDArray[SharedT]:
        """Create a SharedNDArray from a SharedMemoryManager and an existing numpy array.

        Parameters
        ----------
        mem_mgr
            Running `multiprocessing.managers.SharedMemoryManager` instance from which to create the
            SharedMemory for the SharedNDArray
        arr
            NumPy `ndarray` object to copy into the created SharedNDArray upon initialization.
        """
        # Simply use from_shape() to create the SharedNDArray and copy the data into it.
        shared_arr = cls.from_shape(mem_mgr, arr.shape, arr.dtype, lock=lock)
        shared_arr[:] = arr[:]
        return shared_arr

    @classmethod
    def from_shape(
        cls, mem_mgr: SharedMemoryManager, shape: Tuple, dtype: npt.DTypeLike, *, lock: bool = False
    ) -> SharedNDArray:
        """Create a SharedNDArray directly from a SharedMemoryManager

        Parameters
        ----------
        mem_mgr
            SharedMemoryManager instance that has been started
        shape
            Shape of the array
        dtype
            Data type for the NumPy array to be represented in shared memory. Any valid argument for
            `np.dtype` may be used as it will be converted to an actual `dtype` object.
        """
        dtype = np.dtype(dtype)  # Convert to dtype if possible
        shm = mem_mgr.SharedMemory(np.prod(shape) * dtype.itemsize)
        return cls(shm=shm, shape=shape, dtype=dtype, lock=lock)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @shape.setter
    def shape(self, shp: Tuple[int, ...]):
        """Ensure the provided shape is compatible with the data"""
        # Try to reshape using ndarray. Will raise a ValueError if the shape is invalid
        self._shape = self.get().reshape(shp).shape

    def get(self) -> npt.NDArray[SharedT]:
        """Get a numpy array with access to the shared memory"""
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    # FUTURE: Return types cannot currently be expressed with __getitem__ due to dependence on number of
    #         dimensions.
    def __getitem__(self, key: NDSliceLike):
        """Get data from the shared array

        Equivalent to SharedNDArray.get()[key]
        """
        return self.get()[key]

    # FUTURE: Check value dtype?
    def __setitem__(self, key: NDSliceLike, value: npt.ArrayLike):
        """Set values in the shared array,
        Equivalent to SharedNDArray.get()[key] = value
        """
        self.get()[key] = value

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getattr__(self, name):
        extra_err_txt = (
            " To access NumPy ndarray object use .get() method." if hasattr(np.ndarray, name) else ""
        )

        raise AttributeError(f"SharedNDArray object has no attribute '{name}'.{extra_err_txt}")

    def __del__(self):
        if hasattr(self, "shm"):  # Prevent pytest warning...?
            self.shm.close()
