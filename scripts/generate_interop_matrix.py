from pathlib import Path

import h5py
import numpy as np


FILTERS = dict(compression="gzip", shuffle=True, fletcher32=True)


def create_compact(group: h5py.Group, name: str, values: np.ndarray) -> None:
    properties = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    properties.set_layout(h5py.h5d.COMPACT)
    space = h5py.h5s.create_simple(values.shape)
    datatype = h5py.h5t.py_create(values.dtype)
    identifier = h5py.h5d.create(
        group.id, name.encode(), datatype, space, dcpl=properties
    )
    h5py.Dataset(identifier)[...] = values


def populate(handle: h5py.File) -> dict[str, np.ndarray]:
    expected: dict[str, np.ndarray] = {}

    for group_name, count, ordered in (
        ("group_compact", 2, False),
        ("group_dense", 12, False),
        ("group_dense_ordered", 12, True),
    ):
        group = handle.create_group(group_name, track_order=ordered)
        for index in range(count):
            values = np.array([index, index + 100], dtype="<i4")
            group.create_dataset(f"link_{index:02d}", data=values, track_times=False)
            expected[f"{group_name}/link_{index:02d}"] = values

    data = handle.create_group("data", track_order=True)
    create_compact(data, "compact", np.array([30, 40], dtype=">i4"))
    expected["data/compact"] = np.array([30, 40], dtype=">i4")
    data.create_dataset("contiguous", data=np.array([10, 20], dtype="<i4"), track_times=False)
    expected["data/contiguous"] = np.array([10, 20], dtype="<i4")

    raw = np.arange(35, dtype=">i4").reshape(5, 7)
    filtered = (np.arange(35, dtype="<i4") + 100).reshape(5, 7)
    data.create_dataset("chunked_raw", data=raw, chunks=(2, 3), track_times=False)
    expected["data/chunked_raw"] = raw
    data.create_dataset("chunked_filtered", data=filtered, chunks=(2, 3), track_times=False, **FILTERS)
    expected["data/chunked_filtered"] = filtered

    sparse = np.full((7, 9), -7, dtype="<i4")
    sparse[1, 2] = 42
    sparse[6, 8] = 99
    dataset = data.create_dataset(
        "chunked_sparse",
        shape=sparse.shape,
        chunks=(2, 3),
        dtype="<i4",
        fillvalue=-7,
        track_times=False,
    )
    dataset[1, 2] = 42
    dataset[6, 8] = 99
    expected["data/chunked_sparse"] = sparse

    one = np.arange(35, dtype=">i4").reshape(5, 7)
    for name, options in (("unlimited_one", {}), ("unlimited_one_filtered", FILTERS)):
        dataset = data.create_dataset(
            name,
            shape=one.shape,
            maxshape=(None, 7),
            chunks=(2, 3),
            dtype=">i4",
            track_times=False,
            **options,
        )
        dataset[...] = one
        expected[f"data/{name}"] = one

    two = np.arange(100, dtype="<i4").reshape(10, 10)
    for name, options in (("unlimited_two", {}), ("unlimited_two_filtered", FILTERS)):
        dataset = data.create_dataset(
            name,
            shape=two.shape,
            maxshape=(None, None),
            chunks=(2, 2),
            dtype="<i4",
            track_times=False,
            **options,
        )
        dataset[...] = two
        expected[f"data/{name}"] = two

    single = np.arange(12, dtype="<i4").reshape(3, 4)
    for name, options in (("single_raw", {}), ("single_filtered", FILTERS)):
        data.create_dataset(
            name,
            data=single,
            chunks=single.shape,
            track_times=False,
            **options,
        )
        expected[f"data/{name}"] = single

    skipped = data.create_dataset(
        "skipped",
        shape=(8,),
        chunks=(4,),
        dtype="<i4",
        compression="gzip",
        fillvalue=-3,
        track_times=False,
    )
    skipped.id.write_direct_chunk((0,), np.arange(4, dtype="<i4").tobytes(), filter_mask=1)
    expected["data/skipped"] = np.array([0, 1, 2, 3, -3, -3, -3, -3], dtype="<i4")

    for name, count, ordered in (
        ("compact_attrs", 2, False),
        ("compact_attrs_ordered", 2, True),
        ("dense_attrs", 12, False),
        ("dense_attrs_ordered", 12, True),
    ):
        dataset = data.create_dataset(
            name,
            data=np.array([42], dtype="<i4"),
            track_order=ordered,
            track_times=False,
        )
        for index in range(count):
            dataset.attrs[f"attr_{index:02d}"] = index

    return expected


def generate(destination: Path, libver: str | tuple[str, str]) -> None:
    expected = {}
    with h5py.File(destination, "w", libver=libver) as handle:
        expected = populate(handle)
    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
    print(f"Verified {destination.name} with libver {libver} and h5py {h5py.__version__}")


def generate_offset32(destination: Path) -> None:
    creation = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    creation.set_sizes(4, 8)
    access = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    access.set_libver_bounds(h5py.h5f.LIBVER_V110, h5py.h5f.LIBVER_V110)
    identifier = h5py.h5f.create(str(destination).encode(), fcpl=creation, fapl=access)
    with h5py.File(identifier) as handle:
        raw = np.arange(35, dtype=">i4").reshape(5, 7)
        filtered = (np.arange(35, dtype="<i4") + 100).reshape(5, 7)
        handle.create_dataset("raw", data=raw, chunks=(2, 3), track_times=False)
        handle.create_dataset("filtered", data=filtered, chunks=(2, 3), track_times=False, **FILTERS)
    with h5py.File(destination, "r") as handle:
        np.testing.assert_array_equal(handle["raw"][:], raw)
        np.testing.assert_array_equal(handle["filtered"][:], filtered)
    print(f"Verified {destination.name} with 4-byte addresses and h5py {h5py.__version__}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "interop_earliest.h5", "earliest")
    generate(directory / "interop_v108.h5", ("v108", "v108"))
    generate(directory / "interop_v110.h5", ("v110", "v110"))
    generate_offset32(directory / "interop_32.h5")
