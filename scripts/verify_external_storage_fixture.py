import os
import sys
from pathlib import Path

import h5py
import numpy as np


def verify(path: Path) -> None:
    path = path.resolve()
    previous = Path.cwd()
    os.chdir(path.parent)
    try:
        with h5py.File(path.name, "r") as handle:
            multi = handle["multi"]
            assert multi.shape == (8,)
            assert multi.dtype == np.dtype("<i4")
            assert multi.external == [
                ("external_storage_0.bin", 0, 12),
                ("external_storage_1.bin", 0, 20),
            ]
            np.testing.assert_array_equal(
                multi[:], np.arange(8, dtype="<i4")
            )

            offset = handle["offset"]
            assert offset.shape == (4,)
            assert offset.dtype == np.dtype("<i4")
            assert offset.external == [("external_storage_offset.bin", 16, 16)]
            np.testing.assert_array_equal(
                offset[:], np.array([101, 102, 103, 104], dtype="<i4")
            )

            unaligned = handle["unaligned"]
            assert unaligned.shape == (8,)
            assert unaligned.dtype == np.dtype("<i4")
            assert unaligned.external == [
                ("external_storage_unaligned_0.bin", 0, 10),
                ("external_storage_unaligned_1.bin", 0, 22),
            ]
            np.testing.assert_array_equal(
                unaligned[:], np.arange(8, dtype="<i4")
            )

        legacy_path = path.with_name("external_storage_32.h5")
        with h5py.File(legacy_path.name, "r") as handle:
            legacy = handle["legacy"]
            assert legacy.shape == (3,)
            assert legacy.dtype == np.dtype("<i4")
            assert legacy.external == [("external_storage_32.bin", 4, 12)]
            np.testing.assert_array_equal(
                legacy[:], np.array([7, 8, 9], dtype="<i4")
            )
    finally:
        os.chdir(previous)

    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_external_storage_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
