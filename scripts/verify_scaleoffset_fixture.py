from pathlib import Path
import sys

import h5py
import numpy as np


def require_hdf5_1_14_6() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            "scaleoffset fixture verification requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )


def verify(path: Path) -> None:
    require_hdf5_1_14_6()
    expected = {
        "int32_auto": np.concatenate(
            (
                np.arange(-50, -40, dtype="<i4"),
                np.full((10,), -7, dtype="<i4"),
            )
        ),
        "int32_fixed": np.arange(100, 116, dtype="<i4").reshape(4, 4),
        "int32_mixed_fill": np.array([10, -7, 12, 13], dtype="<i4"),
        "int32_all_fill": np.full((4,), -7, dtype="<i4"),
        "int32_full": np.array([-5, -4, -3, -2], dtype="<i4"),
        "int16_be": np.array(
            [-300, -299, -250, -200, -199, -100], dtype=">i2"
        ),
        "int8": np.array([-50, -7, 0, 50], dtype="i1"),
        "uint16_be": np.array([10000, 10001, 15000, 20000], dtype=">u2"),
        "uint8": np.array([0, 1, 2, 3, 4, 5, 6], dtype="u1"),
        "uint32": np.array(
            [1000000, 1000001, 2000000, 3000000], dtype="<u4"
        ),
        "int64": np.array(
            [-(2**40), -(2**40) + 1, 0, 2**40], dtype="<i8"
        ),
        "uint64": np.array(
            [1 << 40, (1 << 40) + 5, (1 << 40) + 9, (1 << 40) + 12],
            dtype="<u8",
        ),
    }
    fills = {
        "int32_auto": -7,
        "int32_fixed": 127,
        "int32_mixed_fill": -7,
        "int32_all_fill": -7,
        "int32_full": 0,
        "int16_be": -1,
        "int8": -7,
        "uint16_be": 42,
        "uint8": 255,
        "uint32": 0,
        "int64": -1,
        "uint64": 7,
    }
    with h5py.File(path, "r") as handle:
        for name, values in expected.items():
            dataset = handle[name]
            np.testing.assert_array_equal(dataset[:], values)
            assert dataset.fillvalue == fills[name]
            properties = dataset.id.get_create_plist()
            assert properties.get_nfilters() == 1
            filter_id, flags, parameters, name_bytes = properties.get_filter(0)
            assert filter_id == 6 and flags == 1
            assert name_bytes == b"scaleoffset"
            assert len(parameters) == 20
            assert parameters[0] == 2
            assert parameters[3] == 0
            assert parameters[4] == dataset.dtype.itemsize
            assert parameters[5] == (1 if dataset.dtype.kind == "i" else 0)
            assert parameters[6] == (
                0
                if dataset.id.get_type().get_order() == h5py.h5t.ORDER_LE
                else 1
            )
            assert parameters[7] == 1
    print(
        f"Verified {path.name} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_scaleoffset_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
