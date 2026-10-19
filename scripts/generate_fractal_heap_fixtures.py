"""Generate dense groups that exercise filtered fractal heaps and huge objects."""

from __future__ import annotations

import ctypes
from pathlib import Path

import h5py
import numpy as np


def load_hdf5() -> ctypes.CDLL:
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and "libhdf5" in fields[-1] and ".so" in fields[-1]:
            library = ctypes.CDLL(fields[-1])
            library.H5Pset_filter.argtypes = [
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_uint,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint),
            ]
            library.H5Pset_filter.restype = ctypes.c_int
            return library
    raise RuntimeError("unable to locate the bundled HDF5 library")


def create_group(handle: h5py.File, name: str, filters: list[tuple[int, list[int]]]) -> h5py.Group:
    properties = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
    library = load_hdf5()
    for filter_id, parameters in filters:
        values = (ctypes.c_uint * len(parameters))(*parameters)
        status = library.H5Pset_filter(
            properties.id,
            filter_id,
            0,
            len(parameters),
            values,
        )
        if status < 0:
            raise RuntimeError(f"unable to add filter {filter_id}")
    return h5py.Group(h5py.h5g.create(handle.id, name.encode(), gcpl=properties))


def add_links(group: h5py.Group, count: int) -> list[str]:
    names = []
    for index in range(count):
        name = f"item_{index:04d}"
        group.create_dataset(name, data=np.array([index], dtype="<i4"), track_times=False)
        names.append(name)
    return names


def generate(destination: Path) -> None:
    long_name = "long_" + "x" * 5000
    with h5py.File(destination, "w", libver="v110") as handle:
        filtered = create_group(handle, "filtered", [(1, [6]), (3, [])])
        filtered_names = add_links(filtered, 1200)
        filtered.create_dataset(long_name, data=np.array([7], dtype="<i4"), track_times=False)
        filtered_names.append(long_name)

        raw = handle.create_group("raw")
        raw_names = add_links(raw, 12)
        raw.create_dataset(long_name, data=np.array([8], dtype="<i4"), track_times=False)
        raw_names.append(long_name)

    with h5py.File(destination, "r") as handle:
        for group_name, expected in (("filtered", filtered_names), ("raw", raw_names)):
            actual = list(handle[group_name].keys())
            np.testing.assert_array_equal(np.array(actual), np.array(expected))
            for index, name in enumerate(expected):
                value = 7 if group_name == "filtered" and name == long_name else (
                    8 if name == long_name else index
                )
                np.testing.assert_array_equal(handle[group_name][name][:], [value])
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "fractal_heap_advanced.h5")
