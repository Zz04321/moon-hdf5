from pathlib import Path
import h5py
import numpy as np


def generate(destination: Path) -> None:
    expected = {}
    with h5py.File(destination, "w", libver=("v110", "v110")) as handle:
        group = handle.create_group("single")
        values = np.arange(12, dtype=">i4").reshape(3, 4)
        for name, options in (("raw", {}), ("filtered", dict(compression="gzip", shuffle=True, fletcher32=True))):
            group.create_dataset(name, data=values, chunks=values.shape, track_times=False, **options)
            expected[f"single/{name}"] = values
        group.create_dataset("unallocated", shape=(3, 4), chunks=(3, 4), dtype="<i4", fillvalue=-7, compression="gzip", track_times=False)
        expected["single/unallocated"] = np.full((3, 4), -7)
        dataset = group.create_dataset("skipped", shape=(4,), chunks=(4,), dtype="<i4", compression="gzip", track_times=False)
        raw = np.array([1, 2, 3, 4], dtype="<i4")
        dataset.id.write_direct_chunk((0,), raw.tobytes(), filter_mask=1)
        expected["single/skipped"] = raw
        group = handle.create_group("implicit")
        for name, shape, maximum in (("edge", (5, 7), (5, 7)), ("resizable", (3, 4), (6, 8))):
            properties = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            properties.set_chunk((2, 3))
            properties.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
            properties.set_fill_value(np.array(-9, dtype="<i4"))
            space = h5py.h5s.create_simple(shape, maximum)
            identifier = h5py.h5d.create(group.id, name.encode(), h5py.h5t.STD_I32LE, space, dcpl=properties)
            dataset = h5py.Dataset(identifier)
            values = np.arange(np.prod(shape), dtype="<i4").reshape(shape)
            dataset[:] = values
            expected[f"implicit/{name}"] = values
        handle.create_dataset("contiguous", data=np.array([10, 20], dtype="<i4"), track_times=False)
        expected["contiguous"] = [10, 20]
        handle.create_dataset("fixed_array", data=np.arange(10, dtype="<i4"), chunks=(3,), track_times=False)
        expected["fixed_array"] = np.arange(10)
        properties = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        properties.set_layout(h5py.h5d.COMPACT)
        identifier = h5py.h5d.create(handle.id, b"compact", h5py.h5t.STD_I32LE, h5py.h5s.create_simple((2,)), dcpl=properties)
        identifier.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([30, 40], dtype="<i4"))
        expected["compact"] = [30, 40]
    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "testdata" / "layout_v4.h5")
