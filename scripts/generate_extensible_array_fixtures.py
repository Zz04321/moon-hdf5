from pathlib import Path

import h5py
import numpy as np


def generate(destination: Path) -> None:
    expected = {}
    with h5py.File(destination, "w", libver=("v110", "v110")) as handle:
        for name, options in (("raw", {}), ("filtered", dict(compression="gzip", shuffle=True, fletcher32=True))):
            group = handle.create_group(name)
            values = np.arange(420, dtype=">i4").reshape(20, 21)
            dataset = group.create_dataset("grown", shape=(0, 21), maxshape=(None, 30), chunks=(2, 3), dtype=">i4", track_times=False, **options)
            dataset.resize(values.shape)
            dataset[:] = values
            expected[f"{name}/grown"] = values
            values = np.arange(210, dtype="<i4").reshape(5, 7, 6)
            group.create_dataset("swizzled", data=values, maxshape=(8, None, 9), chunks=(2, 2, 2), track_times=False, **options)
            expected[f"{name}/swizzled"] = values
            dataset = group.create_dataset("paged", shape=(200000,), maxshape=(None,), chunks=(1,), dtype="<i4", fillvalue=-7, track_times=False, **options)
            values = np.full(200000, -7)
            for index in (0, 3, 4, 19, 20, 51, 52, 115, 116, 243, 244, 499, 500, 1011, 1012, 65523, 65524, 131060, 133108, 199999):
                dataset[index] = index
                values[index] = index
            expected[f"{name}/paged"] = values
        group = handle.create_group("other")
        dataset = group.create_dataset("skipped", shape=(12,), maxshape=(None,), chunks=(4,), dtype="<i4", compression="gzip", fillvalue=-3, track_times=False)
        dataset.id.write_direct_chunk((4,), np.arange(4, dtype="<i4").tobytes(), filter_mask=1)
        expected["other/skipped"] = [-3] * 4 + [0, 1, 2, 3] + [-3] * 4
        group.create_dataset("unallocated", shape=(8,), maxshape=(None,), chunks=(2,), dtype="<i4", fillvalue=-5, track_times=False)
        expected["other/unallocated"] = np.full(8, -5)
        dataset = group.create_dataset("empty", data=np.arange(8, dtype="<i4"), maxshape=(None,), chunks=(2,), track_times=False)
        dataset.resize((0,))
        expected["other/empty"] = []
    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "extensible_array.h5")
    creation = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    creation.set_sizes(4, 8)
    access = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    access.set_libver_bounds(h5py.h5f.LIBVER_V110, h5py.h5f.LIBVER_V110)
    destination = directory / "extensible_array_32.h5"
    identifier = h5py.h5f.create(str(destination).encode(), fcpl=creation, fapl=access)
    values = np.arange(1500, dtype="<i4")
    with h5py.File(identifier) as handle:
        for name, options in (("raw", {}), ("filtered", dict(compression="gzip", shuffle=True, fletcher32=True))):
            handle.create_dataset(name, data=values, chunks=(3,), maxshape=(None,), track_times=False, **options)
    with h5py.File(destination, "r") as handle:
        for name in ("raw", "filtered"):
            np.testing.assert_array_equal(handle[name][:], values)
    print(f"Verified {destination} with 4-byte addresses and 8-byte lengths")
