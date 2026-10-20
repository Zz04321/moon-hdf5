import sys
from pathlib import Path

import h5py
import numpy as np


def verify(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        assert list(handle) == ["experiment"]
        assert sorted(handle["experiment"]) == [
            "chunked",
            "counts",
            "labels",
            "metrics",
        ]
        np.testing.assert_array_equal(
            handle["experiment/counts"][:], [[1, 2, 3], [4, 5, 6]]
        )
        assert handle["experiment/counts"].dtype == np.dtype("<i8")
        assert handle["experiment/labels"].asstr()[:].tolist() == [
            "control",
            "sample",
        ]
        np.testing.assert_array_equal(
            handle["experiment/metrics"][:], [1.5, -2.25]
        )
        assert handle["experiment/metrics"].dtype == np.dtype("<f8")
        assert handle["experiment"].attrs["revision"] == np.uint64(1)
        chunked = handle["experiment/chunked"]
        expected = np.full((5, 7), -7, dtype="<i8")
        expected[0, 0] = 42
        expected[2, 3] = 99
        expected[4, 6] = -2
        np.testing.assert_array_equal(chunked[:], expected)
        assert chunked.chunks == (2, 3)
        assert chunked.fillvalue == -7
        assert chunked.shuffle
        assert chunked.compression == "gzip"
        assert chunked.compression_opts == 6
        assert chunked.fletcher32
        unit = handle["experiment/counts"].attrs["unit"]
        assert unit.decode("utf-8").rstrip("\0") == "items"
    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_writer_output.py <file.h5>")
    verify(Path(sys.argv[1]))
