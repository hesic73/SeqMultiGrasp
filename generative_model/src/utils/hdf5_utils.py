import numpy as np

from collections import defaultdict
import h5py
import os

from typing import Dict, Any


def append_to_hdf5(filename: str, data: Dict[str, np.ndarray]):
    """
    Append a batch of data to an HDF5 file.

    Parameters:
    - filename (str): The HDF5 file to write to. If it doesn't exist, it will be created.
    - data (dict[str, np.ndarray]): A dictionary where each key is a dataset name and
      the value is a NumPy array. All arrays must have the same size along the first dimension.
    """
    # Validation
    if not isinstance(data, dict) or not all(isinstance(v, np.ndarray) for v in data.values()):
        raise ValueError("Input data must be a dictionary of NumPy arrays.")

    batch_size = None
    for key, array in data.items():
        if batch_size is None:
            batch_size = array.shape[0]
        elif array.shape[0] != batch_size:
            raise ValueError(
                f"Inconsistent batch sizes: {key} has size {array.shape[0]}, expected {batch_size}.")

    with h5py.File(filename, "a") as f:
        for key, array in data.items():
            # Check if dataset exists
            if key not in f:
                maxshape = (None,) + array.shape[1:]
                dtype = h5py.string_dtype(
                    encoding="utf-8") if array.dtype.kind == "U" else array.dtype
                f.create_dataset(key, shape=(
                    0,) + array.shape[1:], maxshape=maxshape, dtype=dtype)

            # Append data
            dset = f[key]
            current_size = dset.shape[0]
            new_size = current_size + batch_size
            dset.resize((new_size,) + dset.shape[1:])
            dset[current_size:new_size] = array


def create_hdf5(filename: str, data: Dict[str, np.ndarray], mode: str = "w"):
    """
    Create an HDF5 file and write datasets to it.

    Parameters:
    - filename (str): The HDF5 file to write to.
    - data (dict[str, np.ndarray]): A dictionary where each key is a dataset name and
      the value is a NumPy array. All arrays must have the same size along the first dimension.
    - mode (str): File mode ('w' to overwrite the file, 'x' to create a new file and fail if it exists).
      - 'w': Create file, truncate if it exists (overwrite entire file).
      - 'x': Create file, fail if it exists.
    """
    if mode not in {"w", "x"}:
        raise ValueError(
            "Invalid mode. Use 'w' to overwrite or 'x' to create a new file.")

    # Validation
    if not isinstance(data, dict) or not all(isinstance(v, np.ndarray) for v in data.values()):
        raise ValueError("Input data must be a dictionary of NumPy arrays.")

    # Open the file with the specified mode
    with h5py.File(filename, mode) as f:
        for key, array in data.items():

            maxshape = (None,) + array.shape[1:]
            dtype = h5py.string_dtype(
                encoding="utf-8") if array.dtype.kind == "U" else array.dtype
            f.create_dataset(key, shape=array.shape,
                             maxshape=maxshape, dtype=dtype)

            # write data
            dset = f[key]
            dset[:] = array


class HDF5Buffer:
    def __init__(self, filename: str, buffer_size: int = 1000, exists_ok: bool = False):
        """
        Initialize an HDF5Buffer instance.

        Parameters:
        - filename (str): Path to the HDF5 file.
        - buffer_size (int): Maximum size of the buffer before flushing.
        - exists_ok (bool): If False, an exception will be raised if the file already exists.
        """
        if not exists_ok and os.path.exists(filename):
            raise ValueError(f"Output path {filename} already exists!")

        self.filename = filename
        self.buffer_size = buffer_size
        self.buffer = defaultdict(list)

    def append(self, data: Dict[str, Any]):
        """
        Append data to the buffer.

        Parameters:
        - data (dict): A dictionary where each key is a dataset name and the value is a NumPy array.
        """
        for key, value in data.items():
            self.buffer[key].append(value)

        # Check if buffer size exceeds the threshold
        if len(next(iter(self.buffer.values()))) >= self.buffer_size:
            self.flush()

    def flush(self):
        """
        Write the current buffer to the HDF5 file and clear the buffer.
        """
        if not self.buffer:
            return

        # Combine lists into NumPy arrays
        data_to_save = {key: np.array(values)
                        for key, values in self.buffer.items()}

        append_to_hdf5(self.filename, data_to_save)

        # Clear the buffer
        self.buffer.clear()

    def close(self):
        """
        Flush any remaining data in the buffer and close the HDF5 file.
        """
        self.flush()
