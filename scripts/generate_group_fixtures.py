from pathlib import Path

import h5py
import numpy as np


def generate(destination: Path) -> None:
    with h5py.File(destination, "w", libver=("v108", "v108")) as handle:
        handle.create_group("empty")
        for name, count, ordered in (
            ("compact", 2, False),
            ("compact_ordered", 2, True),
            ("dense", 12, False),
            ("dense_ordered", 12, True),
        ):
            group = handle.create_group(name, track_order=ordered)
            for index in range(count):
                group.create_dataset(
                    f"value_{index:02d}",
                    data=np.array([index, index + 100], dtype="<i4"),
                    track_times=False,
                )
    with h5py.File(destination, "r") as handle:
        assert len(handle["empty"]) == 0
        for name, count in (
            ("compact", 2),
            ("compact_ordered", 2),
            ("dense", 12),
            ("dense_ordered", 12),
        ):
            assert list(handle[name]) == [f"value_{index:02d}" for index in range(count)]
            for index in range(count):
                np.testing.assert_array_equal(
                    handle[name][f"value_{index:02d}"][:], [index, index + 100]
                )
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "testdata" / "group_storage.h5")
