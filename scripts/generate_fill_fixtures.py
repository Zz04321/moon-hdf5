from pathlib import Path

import h5py
import numpy as np


def generate(destination: Path, bounds: tuple[str, str]) -> None:
    expected = {}
    with h5py.File(destination, "w", libver=bounds) as handle:
        handle.create_group("numbers")
        handle.create_group("other")
        for name, dtype, fill in (
            ("signed", "<i4", -7),
            ("big_endian", ">i4", -1234),
            ("floating", "<f8", 1.5),
            ("text", "S4", b"none"),
        ):
            name = ("other/" if name == "text" else "numbers/") + name
            dataset = handle.create_dataset(
                name, shape=(3, 5), chunks=(2, 3), dtype=dtype,
                fillvalue=fill, track_times=False,
            )
            values = np.full((3, 5), fill, dtype=dtype)
            values[0, 0] = b"yes" if name == "other/text" else 42
            values[2, 4] = b"end" if name == "other/text" else 99
            dataset[0, 0] = values[0, 0]
            dataset[2, 4] = values[2, 4]
            assert dataset.id.get_num_chunks() == 2
            expected[name] = values
        for name, fill in (("unallocated", -9), ("default", None)):
            name = "other/" + name
            dataset = handle.create_dataset(
                name, shape=(4,), chunks=(2,), dtype="<i4",
                fillvalue=fill, track_times=False,
            )
            assert dataset.id.get_storage_size() == 0
            expected[name] = np.full(4, 0 if fill is None else fill, dtype="<i4")
        dataset = handle.create_dataset(
            "other/partial_default", shape=(4,), chunks=(2,), dtype="<i4",
            track_times=False,
        )
        dataset[0] = 5
        expected["other/partial_default"] = np.array([5, 0, 0, 0], dtype="<i4")
    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "fill_v2.h5", ("earliest", "v108"))
    generate(directory / "fill_v3.h5", ("v108", "v108"))
