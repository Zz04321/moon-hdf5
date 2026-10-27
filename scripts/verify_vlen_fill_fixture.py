from pathlib import Path
import sys

import h5py


def require_hdf5_1_14_6() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            "variable-length fill fixture verification requires HDF5 "
            "1.14.6, found "
            + h5py.version.hdf5_version
        )


def verify(path: Path) -> None:
    require_hdf5_1_14_6()
    expected = {
        "contiguous": ["missing", "missing", "missing", "missing"],
        "chunked": [
            "first",
            "fallback",
            "fallback",
            "fallback",
            "fallback",
            "last",
        ],
        "extensible": ["grow", "grow", "grow", "grow", "grow"],
    }
    # HDF5 1.14.6 needs write intent to materialize a vlen fill from an
    # unallocated chunk; MoonBit reads the same file from immutable bytes.
    with h5py.File(path, "r+") as handle:
        for name, values in expected.items():
            dataset = handle[name]
            assert h5py.check_string_dtype(dataset.dtype).encoding == "utf-8"
            assert dataset.asstr()[:].tolist() == values
        assert handle["contiguous"].fillvalue == b"missing"
        assert handle["chunked"].fillvalue == b"fallback"
        assert handle["chunked"].id.get_num_chunks() == 2
        assert handle["extensible"].maxshape == (None,)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_vlen_fill_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
