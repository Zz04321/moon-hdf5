from pathlib import Path

import h5py
import numpy as np


def generate(destination: Path) -> None:
    expected = {}
    with h5py.File(destination, "w", libver=("v110", "v110")) as handle:
        for name, options in (("raw", {}), ("filtered", dict(compression="gzip", shuffle=True, fletcher32=True))):
            group = handle.create_group(name)
            values = np.arange(35, dtype=">i4").reshape(5, 7)
            dataset = group.create_dataset("edge", shape=(0, 0), maxshape=(None, None), chunks=(2, 3), dtype=">i4", track_times=False, **options)
            dataset.resize(values.shape)
            dataset[:] = values
            expected[f"{name}/edge"] = values
            values = np.arange(10000, dtype="<i4").reshape(100, 100)
            group.create_dataset("deep", data=values, maxshape=(None, None), chunks=(1, 1), track_times=False, **options)
            expected[f"{name}/deep"] = values
            values = np.full((7, 9), -7, dtype="<i4")
            values[1, 2] = 42
            values[6, 8] = 99
            dataset = group.create_dataset("sparse", shape=(7, 9), maxshape=(None, None), chunks=(2, 3), dtype="<i4", fillvalue=-7, track_times=False, **options)
            dataset[1, 2] = 42
            dataset[6, 8] = 99
            expected[f"{name}/sparse"] = values
        group = handle.create_group("other")
        dataset = group.create_dataset("skipped", shape=(2, 4), maxshape=(None, None), chunks=(1, 4), dtype="<i4", fillvalue=-3, compression="gzip", track_times=False)
        dataset.id.write_direct_chunk((0, 0), np.arange(4, dtype="<i4").tobytes(), filter_mask=1)
        expected["other/skipped"] = [[0, 1, 2, 3], [-3, -3, -3, -3]]
        group.create_dataset("unallocated", shape=(2, 4), maxshape=(None, None), chunks=(1, 2), dtype="<i4", fillvalue=-5, track_times=False)
        expected["other/unallocated"] = np.full((2, 4), -5)
        dataset = group.create_dataset("empty", data=np.arange(8, dtype="<i4").reshape(2, 4), maxshape=(None, None), chunks=(1, 2), track_times=False)
        dataset.resize((0, 4))
        expected["other/empty"] = np.empty((0, 4))
    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "testdata" / "btree_v2.h5")
