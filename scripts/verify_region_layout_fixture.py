"""Verify the 4-byte HDF5 1.14.6 region-reference layout fixture."""

from pathlib import Path
import sys

import h5py
import numpy as np


def read_unsigned(data: bytes, offset: int, size: int) -> int:
    return int.from_bytes(data[offset : offset + size], "little")


def read_heap_payload(path: Path, reference: bytes) -> bytes:
    data = path.read_bytes()
    offset_size = data[9]
    length_size = data[10]
    heap_address = read_unsigned(reference, 0, offset_size)
    object_index = read_unsigned(reference, offset_size, 4)
    assert data[heap_address : heap_address + 4] == b"GCOL"
    collection_size = read_unsigned(
        data, heap_address + 8, length_size
    )
    assert heap_address + collection_size <= len(data)
    cursor = heap_address + 8 + length_size
    object_header_size = 8 + length_size
    while cursor + object_header_size <= heap_address + collection_size:
        index = read_unsigned(data, cursor, 2)
        size = read_unsigned(data, cursor + 8, length_size)
        cursor += object_header_size
        if index == 0:
            break
        assert cursor + size <= heap_address + collection_size
        if index == object_index:
            return data[cursor : cursor + size]
        cursor += (size + 7) // 8 * 8
    raise AssertionError(f"heap object {object_index} not found")


def decode_selection(payload: bytes, offset_size: int) -> dict:
    target_address = read_unsigned(payload, 0, offset_size)
    cursor = offset_size
    selection_type = read_unsigned(payload, cursor, 4)
    version = read_unsigned(payload, cursor + 4, 4)
    cursor += 8
    if selection_type == 1:
        assert version == 2
        encoding_size = payload[cursor]
        cursor += 1
        rank = read_unsigned(payload, cursor, 4)
        count = read_unsigned(payload, cursor + 4, encoding_size)
        cursor += 4 + encoding_size
        points = []
        for _ in range(count):
            point = []
            for _ in range(rank):
                point.append(read_unsigned(payload, cursor, encoding_size))
                cursor += encoding_size
            points.append(point)
        assert cursor == len(payload)
        return {
            "target_address": target_address,
            "selection": ("points", points),
        }
    assert selection_type == 2 and version == 3
    flags = payload[cursor]
    encoding_size = payload[cursor + 1]
    cursor += 2
    rank = read_unsigned(payload, cursor, 4)
    cursor += 4
    assert flags & 1
    vectors = [[], [], [], []]
    for _ in range(rank):
        for vector in vectors:
            vector.append(read_unsigned(payload, cursor, encoding_size))
            cursor += encoding_size
    assert cursor == len(payload)
    return {
        "target_address": target_address,
        "selection": ("hyperslab", *vectors),
    }


def verify_serialized_references(path: Path) -> None:
    expected = {
        "ref_compact": (
            "compact",
            {
                "selection": (
                    "hyperslab",
                    [1],
                    [2],
                    [2],
                    [1],
                )
            },
        ),
        "ref_contiguous": (
            "contiguous",
            {"selection": ("points", [[4], [0], [2]])},
        ),
        "ref_chunked": (
            "chunked",
            {
                "selection": (
                    "hyperslab",
                    [0],
                    [3],
                    [2],
                    [2],
                )
            },
        ),
    }
    data = path.read_bytes()
    offset_size = data[9]
    length_size = data[10]
    assert offset_size == 4
    assert length_size == 8
    with h5py.File(path, "r", libver="latest") as handle:
        for reference_name, (target_name, selection) in expected.items():
            reference_dataset = handle[reference_name]
            assert reference_dataset.id.get_storage_size() == offset_size + 4
            raw = data[
                reference_dataset.id.get_offset() : reference_dataset.id.get_offset()
                + reference_dataset.id.get_storage_size()
            ]
            payload = read_heap_payload(path, raw)
            decoded = decode_selection(payload, offset_size)
            target = handle[target_name]
            assert decoded["target_address"] == h5py.h5o.get_info(target.id).addr
            assert decoded["selection"] == selection["selection"]
            reference = reference_dataset[0]
            assert bool(reference)
            assert handle[reference].name == target.name


def verify(path: Path) -> None:
    layouts = {
        h5py.h5d.COMPACT: "compact",
        h5py.h5d.CONTIGUOUS: "contiguous",
        h5py.h5d.CHUNKED: "chunked",
    }

    with h5py.File(path, "r", libver="latest") as handle:
        assert handle.id.get_create_plist().get_sizes() == (4, 8)
        for name, layout in (
            ("compact", "compact"),
            ("contiguous", "contiguous"),
            ("chunked", "chunked"),
        ):
            actual = layouts[handle[name].id.get_create_plist().get_layout()]
            assert actual == layout
        assert handle["chunked"].shuffle
        assert handle["chunked"].compression == "gzip"
        assert handle["chunked"].chunks == (2,)
        for name in ("compact", "contiguous", "chunked"):
            np.testing.assert_array_equal(
                handle[name][...], np.arange(6, dtype="<i4")
            )
    verify_serialized_references(path)

    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_region_layout_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
