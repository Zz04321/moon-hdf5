from pathlib import Path

import h5py
import numpy as np


def generate(destination: Path, bounds: tuple[str, str]) -> None:
    expected = {}
    with h5py.File(destination, "w", libver=bounds) as handle:
        for name, dtype, fill in (
            ("signed", "<i4", -7), ("big_endian", ">i4", -1234),
            ("unsigned", ">u8", 2**63 + 1), ("floating", ">f8", 1.5),
            ("text", "S4", b"none"), ("default", "<i4", None),
        ):
            dataset = handle.create_dataset(
                name, shape=(3,), dtype=dtype, fillvalue=fill, track_times=False,
            )
            assert dataset.chunks is None and dataset.id.get_offset() is None
            expected[name] = np.full(3, 0 if fill is None else fill, dtype=dtype)
        group = handle.create_group("other")
        group.create_dataset("scalar", shape=(), dtype="<i4", fillvalue=-2, track_times=False)
        expected["other/scalar"] = np.array(-2, dtype="<i4")
        group.create_dataset("empty", shape=(0, 3), dtype="<i4", track_times=False)
        expected["other/empty"] = np.empty((0, 3), dtype="<i4")
        group.create_dataset("null", dtype="<i4", track_times=False)
        dataset = group.create_dataset("allocated", shape=(3,), dtype="<i4", fillvalue=-7, track_times=False)
        dataset[1] = 42
        expected["other/allocated"] = np.array([-7, 42, -7], dtype="<i4")
    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][()], values)
        assert handle["other/null"].shape is None
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "contiguous_fill_v2.h5", ("earliest", "v108"))
    generate(directory / "contiguous_fill_v3.h5", ("v108", "v108"))
