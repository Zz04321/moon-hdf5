from pathlib import Path

import h5py
import numpy as np


def enum_dtypes() -> tuple[np.dtype, np.dtype, np.dtype]:
    signed = h5py.enum_dtype(
        {
            "RED": 0,
            "GREEN": 1,
            "LONG_NAME_XYZ": 2,
            "EIGHTCHR": 3,
        },
        basetype=np.dtype("<i4"),
    )
    unsigned = h5py.enum_dtype(
        {"ZERO": 0, "ONE": 1, "LONG_NAME_XYZ": 2},
        basetype=np.dtype("<u1"),
    )
    big_endian = h5py.enum_dtype(
        {"LOW": -1, "HIGH": 300},
        basetype=np.dtype(">i2"),
    )
    return signed, unsigned, big_endian


def generate(destination: Path, bounds: str | tuple[str, str]) -> None:
    signed, unsigned, big_endian = enum_dtypes()
    with h5py.File(destination, "w", libver=bounds) as handle:
        handle.create_dataset(
            "colors",
            data=np.array([0, 2, 1, 3], dtype="<i4"),
            dtype=signed,
            track_times=False,
        )
        handle.create_dataset(
            "unsigned",
            data=np.array([2, 0], dtype="<u1"),
            dtype=unsigned,
            track_times=False,
        )
        handle.create_dataset(
            "big_endian",
            data=np.array([300, -1], dtype=">i2"),
            dtype=big_endian,
            track_times=False,
        )
        dataset = handle["colors"]
        dataset.attrs.create("favorite", np.array(1, dtype="<i4"), dtype=signed)
        dataset.attrs.create("pair", np.array([2, 0], dtype="<i4"), dtype=signed)

    with h5py.File(destination, "r") as handle:
        np.testing.assert_array_equal(handle["colors"][:], [0, 2, 1, 3])
        np.testing.assert_array_equal(handle["unsigned"][:], [2, 0])
        np.testing.assert_array_equal(handle["big_endian"][:], [300, -1])
        assert handle["colors"].attrs["favorite"] == 1
        np.testing.assert_array_equal(handle["colors"].attrs["pair"], [2, 0])
    print(f"Verified {destination.name} with libver {bounds} and h5py {h5py.__version__}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "enums_v108.h5", ("earliest", "v108"))
    generate(directory / "enums_v110.h5", ("v110", "v110"))
