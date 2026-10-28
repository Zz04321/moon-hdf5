import sys
from pathlib import Path

import h5py
import numpy as np


def verify(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        assert list(handle) == [
            "counts",
            "empty",
            "enabled",
            "label",
            "record",
            "temperature",
        ]
        assert handle["temperature"].attrs["MATLAB_class"] == b"double"
        np.testing.assert_array_equal(
            handle["temperature"][:],
            np.array([[21.5, 23.25], [22.0, -4.5]], dtype="<f8"),
        )
        assert handle["counts"].attrs["MATLAB_class"] == b"int32"
        np.testing.assert_array_equal(
            handle["counts"][:],
            np.array([[0, 3], [1, 4], [2, 5]], dtype="<i4"),
        )
        assert handle["enabled"].attrs["MATLAB_class"] == b"logical"
        assert handle["enabled"].attrs["MATLAB_int_decode"] == 1
        np.testing.assert_array_equal(
            handle["enabled"][:], np.array([[1, 0], [0, 1]], dtype=np.uint8)
        )
        assert handle["label"].attrs["MATLAB_class"] == b"char"
        assert handle["label"].attrs["MATLAB_int_decode"] == 2
        np.testing.assert_array_equal(
            handle["label"][:], np.array([[28201], [24230]], dtype=np.uint16)
        )
        assert handle["record"].attrs["MATLAB_class"] == b"struct"
        assert len(handle["record"].attrs["MATLAB_fields"]) == 2
        assert handle["record/count"].attrs["MATLAB_class"] == b"uint16"
        np.testing.assert_array_equal(handle["record/count"][:], [[42]])
        assert handle["record/samples"].attrs["MATLAB_class"] == b"uint8"
        np.testing.assert_array_equal(handle["record/samples"][:], [[1], [2], [3]])
        assert handle["empty"].attrs["MATLAB_class"] == b"double"
        assert handle["empty"].attrs["MATLAB_empty"] == 1
        np.testing.assert_array_equal(handle["empty"][:], [1, 0])
    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_matlab73_fixture.py <file.mat>")
    verify(Path(sys.argv[1]))
