from pathlib import Path

import h5py
import numpy as np


def generate(destination: Path) -> None:
    expected = {}
    with h5py.File(destination, "w", libver=("v110", "v110")) as handle:
        for name, options in (("raw", {}), ("filtered", dict(compression="gzip", shuffle=True, fletcher32=True))):
            values = np.arange(35, dtype=">i4").reshape(5, 7)
            handle.create_dataset(name, data=values, chunks=(2, 3), track_times=False, **options)
            expected[name] = values
        dataset = handle.create_dataset("resizable", shape=(3, 4), maxshape=(6, 8), chunks=(2, 3), dtype="<i4", fillvalue=-9, track_times=False)
        dataset[:] = np.arange(12).reshape(3, 4)
        expected["resizable"] = np.arange(12).reshape(3, 4)
        for name, options in (("paged", {}), ("paged_filtered", dict(compression="gzip", shuffle=True, fletcher32=True))):
            dataset = handle.create_dataset(name, shape=(4099,), chunks=(1,), dtype="<i4", fillvalue=-7, track_times=False, **options)
            values = np.full(4099, -7)
            for index in (0, 1023, 2048, 4098):
                dataset[index] = index
                values[index] = index
            expected[name] = values
        dataset = handle.create_dataset("skipped", shape=(8,), chunks=(4,), dtype="<i4", compression="gzip", fillvalue=-3, track_times=False)
        dataset.id.write_direct_chunk((0,), np.arange(4, dtype="<i4").tobytes(), filter_mask=1)
        expected["skipped"] = [0, 1, 2, 3, -3, -3, -3, -3]
        handle.create_dataset("unallocated", shape=(8,), chunks=(2,), dtype="<i4", compression="gzip", fillvalue=-5, track_times=False)
        expected["unallocated"] = np.full(8, -5)
    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "testdata" / "fixed_array.h5")
