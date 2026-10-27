from pathlib import Path
import os

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"


def generate() -> None:
    previous = Path.cwd()
    os.chdir(TESTDATA)
    try:
        for name in (
            "external_storage.h5",
            "external_storage_0.bin",
            "external_storage_1.bin",
            "external_storage_offset.bin",
            "external_storage_unaligned_0.bin",
            "external_storage_unaligned_1.bin",
            "external_storage_32.h5",
            "external_storage_32.bin",
        ):
            path = TESTDATA / name
            if path.exists():
                path.unlink()

        values = np.arange(8, dtype="<i4")
        offset_values = np.array([101, 102, 103, 104], dtype="<i4")
        unaligned_values = np.arange(8, dtype="<i4")
        with h5py.File("external_storage.h5", "w", libver=("v110", "v110")) as handle:
            multi = handle.create_dataset(
                "multi",
                shape=values.shape,
                dtype=values.dtype,
                external=[
                    ("external_storage_0.bin", 0, 12),
                    ("external_storage_1.bin", 0, 20),
                ],
            )
            multi[...] = values
            offset = handle.create_dataset(
                "offset",
                shape=offset_values.shape,
                dtype=offset_values.dtype,
                external=[("external_storage_offset.bin", 16, 16)],
            )
            offset[...] = offset_values
            unaligned = handle.create_dataset(
                "unaligned",
                shape=unaligned_values.shape,
                dtype=unaligned_values.dtype,
                external=[
                    ("external_storage_unaligned_0.bin", 0, 10),
                    ("external_storage_unaligned_1.bin", 0, 22),
                ],
            )
            unaligned[...] = unaligned_values

        with h5py.File("external_storage.h5", "r") as handle:
            np.testing.assert_array_equal(handle["multi"][...], values)
            np.testing.assert_array_equal(handle["offset"][...], offset_values)
            np.testing.assert_array_equal(
                handle["unaligned"][...], unaligned_values
            )

        legacy_values = np.array([7, 8, 9], dtype="<i4")
        with h5py.File("external_storage_32.h5", "w", libver="earliest") as handle:
            legacy = handle.create_dataset(
                "legacy",
                shape=legacy_values.shape,
                dtype=legacy_values.dtype,
                external=[("external_storage_32.bin", 4, 12)],
            )
            legacy[...] = legacy_values

        with h5py.File("external_storage_32.h5", "r") as handle:
            np.testing.assert_array_equal(handle["legacy"][...], legacy_values)
    finally:
        os.chdir(previous)
    print(
        "Verified external raw-data fixture with h5py "
        f"{h5py.__version__}/HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    generate()
