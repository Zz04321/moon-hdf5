"""Generate a 1.14.6 fixture for SOHM and internally shared attributes."""

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
            library.H5Pset_shared_mesg_nindexes.argtypes = [
                ctypes.c_long,
                ctypes.c_uint,
            ]
            library.H5Pset_shared_mesg_nindexes.restype = ctypes.c_int
            library.H5Pset_shared_mesg_index.argtypes = [
                ctypes.c_long,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
            ]
            library.H5Pset_shared_mesg_index.restype = ctypes.c_int
            library.H5Pset_shared_mesg_phase_change.argtypes = [
                ctypes.c_long,
                ctypes.c_uint,
                ctypes.c_uint,
            ]
            library.H5Pset_shared_mesg_phase_change.restype = ctypes.c_int
            return library
    raise RuntimeError("unable to locate the bundled HDF5 library")


def check(status: int, operation: str) -> None:
    if status < 0:
        raise RuntimeError(f"unable to {operation}")


def create_file(destination: Path) -> None:
    library = load_hdf5()
    creation = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    check(library.H5Pset_shared_mesg_nindexes(creation.id, 3), "set indexes")
    check(
        library.H5Pset_shared_mesg_index(creation.id, 0, 0x000A, 0),
        "share datatype and dataspace messages",
    )
    # Small attributes stay inline so their datatype and dataspace fields can
    # be shared independently. Larger attributes also share the whole message.
    check(
        library.H5Pset_shared_mesg_index(creation.id, 1, 0x1000, 50),
        "share attribute messages",
    )
    check(
        library.H5Pset_shared_mesg_index(creation.id, 2, 0x0020, 0),
        "share fill-value messages",
    )
    check(
        library.H5Pset_shared_mesg_phase_change(creation.id, 4, 2),
        "set shared-message phase change",
    )
    file_id = h5py.h5f.create(
        str(destination).encode(), h5py.h5f.ACC_TRUNC, fcpl=creation
    )
    file_id.close()


def add_cases(destination: Path) -> None:
    small = np.array([7, 42], dtype="<i4")
    large = np.arange(16, dtype="<i4")
    dense_common = large + 7
    with h5py.File(destination, "a") as handle:
        handle.attrs["rootattr"] = small
        values = np.arange(4, dtype="<i4")
        for index in range(12):
            dataset = handle.create_dataset(
                f"d{index}", data=values, track_times=False
            )
            dataset.attrs["common"] = small

        for index in range(12):
            dataset = handle.create_dataset(
                f"large{index}", data=values, track_times=False
            )
            dataset.attrs["large_common"] = large

        for index in range(4):
            handle.create_dataset(
                f"fill{index}",
                shape=(4,),
                dtype="<i4",
                fillvalue=-7,
                track_times=False,
            )

        for index in range(6):
            group = handle.create_group(f"component{index}")
            group.attrs.create(f"unique{index}", small)

        for name in ("dense0", "dense1"):
            group = handle.create_group(name)
            group.attrs.create("common", dense_common)
            for index in range(12):
                group.attrs.create(
                    f"{name}_unique{index:02d}", large + index
                )

        datatype = h5py.h5t.py_create(np.dtype("<i4"), logical=True)
        datatype.commit(handle.id, b"named_i4")
        group = handle.create_group("committed")
        attribute = h5py.h5a.create(
            group.id,
            b"named",
            datatype,
            h5py.h5s.create_simple((2,)),
        )
        attribute.write(small)
        dataset = h5py.h5d.create(
            handle.id,
            b"committed_value",
            datatype,
            h5py.h5s.create_simple((3,)),
        )
        dataset.write(
            h5py.h5s.ALL,
            h5py.h5s.ALL,
            np.array([4, 5, 6], dtype="<i4"),
        )

        handle.create_dataset(
            "chunked",
            data=np.arange(20, dtype="<i4").reshape(4, 5),
            chunks=(2, 3),
            track_times=False,
        )


def verify(destination: Path) -> None:
    small = np.array([7, 42], dtype="<i4")
    large = np.arange(16, dtype="<i4")
    dense_common = large + 7
    with h5py.File(destination, "r") as handle:
        np.testing.assert_array_equal(handle.attrs["rootattr"], small)
        for index in range(12):
            np.testing.assert_array_equal(handle[f"d{index}"][:], [0, 1, 2, 3])
            np.testing.assert_array_equal(handle[f"d{index}"].attrs["common"], small)
            np.testing.assert_array_equal(
                handle[f"large{index}"].attrs["large_common"], large
            )
        for index in range(4):
            np.testing.assert_array_equal(handle[f"fill{index}"][:], [-7] * 4)
        for index in range(6):
            np.testing.assert_array_equal(
                handle[f"component{index}"].attrs[f"unique{index}"], small
            )
        for name in ("dense0", "dense1"):
            np.testing.assert_array_equal(
                handle[name].attrs["common"], dense_common
            )
            for index in range(12):
                np.testing.assert_array_equal(
                    handle[name].attrs[f"{name}_unique{index:02d}"], large + index
                )
        np.testing.assert_array_equal(handle["committed"].attrs["named"], small)
        np.testing.assert_array_equal(handle["committed_value"][:], [4, 5, 6])
        np.testing.assert_array_equal(
            handle["chunked"][:], np.arange(20).reshape(4, 5)
        )
    data = destination.read_bytes()
    if b"SMTB" not in data or data.count(b"FRHP") < 3:
        raise RuntimeError("fixture does not contain the expected SOHM structures")


def generate(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            "this fixture is pinned to HDF5 1.14.6; found "
            + h5py.version.hdf5_version
        )
    destination.unlink(missing_ok=True)
    create_file(destination)
    add_cases(destination)
    verify(destination)
    print(
        f"Verified {destination} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "shared_messages.h5")
