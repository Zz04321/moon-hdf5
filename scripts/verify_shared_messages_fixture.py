import sys
from pathlib import Path

import h5py
import numpy as np


SHARED_FLAG = 0x02


def uint(data: bytes, offset: int, size: int) -> int:
    if size <= 0 or offset < 0 or offset + size > len(data):
        raise AssertionError("fixture field is outside the file")
    return int.from_bytes(data[offset : offset + size], "little")


def object_header_messages(data: bytes, address: int) -> list[tuple[int, int, bytes]]:
    assert data[address : address + 4] == b"OHDR", "expected a v2 object header"
    flags = data[address + 5]
    offset = address + 6
    if flags & 0x20:
        offset += 16
    if flags & 0x10:
        offset += 4
    chunk_size_width = 1 << (flags & 3)
    chunk_size = uint(data, offset, chunk_size_width)
    block_start = offset + chunk_size_width
    blocks = [(block_start, chunk_size)]
    messages = []
    while blocks:
        start, size = blocks.pop(0)
        end = start + size
        assert end <= len(data), "object header block is truncated"
        offset = start
        message_header_size = 6 if flags & 0x04 else 4
        while offset < end:
            message_type = data[offset]
            message_size = uint(data, offset + 1, 2)
            message_flags = data[offset + 3]
            offset += message_header_size
            assert offset + message_size <= end, "object header message is truncated"
            payload = data[offset : offset + message_size]
            offset += message_size
            messages.append((message_type, message_flags, payload))
            if message_type == 0x10:
                continuation_address = uint(payload, 0, 8)
                continuation_length = uint(payload, 8, 8)
                assert data[continuation_address : continuation_address + 4] == b"OCHK"
                assert continuation_length >= 8
                blocks.append(
                    (continuation_address + 4, continuation_length - 8)
                )
    return messages


def dense_attribute_records(data: bytes, address: int) -> list[bytes]:
    messages = object_header_messages(data, address)
    infos = [payload for kind, _, payload in messages if kind == 0x15]
    assert len(infos) == 1, "expected one attribute-info message"
    info = infos[0]
    assert info[0] == 0, "unsupported attribute-info version"
    flags = info[1]
    offset = 2 + (2 if flags & 1 else 0)
    _heap_address = uint(info, offset, 8)
    offset += 8
    name_index_address = uint(info, offset, 8)
    assert name_index_address != 0xFFFFFFFFFFFFFFFF

    header = name_index_address
    assert data[header : header + 4] == b"BTHD"
    assert data[header + 4] == 0
    assert data[header + 5] == 8, "attribute name B-tree has the wrong type"
    node_size = uint(data, header + 6, 4)
    record_size = uint(data, header + 10, 2)
    depth = uint(data, header + 12, 2)
    root_address = uint(data, header + 16, 8)
    root_count = uint(data, header + 24, 2)
    total = uint(data, header + 26, 8)
    assert record_size == 17 and depth == 0 and total == root_count
    assert data[root_address : root_address + 4] == b"BTLF"
    assert data[root_address + 5] == 8
    start = root_address + 6
    assert start + root_count * record_size <= root_address + node_size
    return [
        data[start + index * record_size : start + (index + 1) * record_size]
        for index in range(root_count)
    ]


def verify(path: Path) -> None:
    small = np.array([7, 42], dtype="<i4")
    large = np.arange(16, dtype="<i4")
    dense_common = large + 7
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle.attrs["rootattr"], small)
        for index in range(12):
            np.testing.assert_array_equal(handle[f"d{index}"][:], [0, 1, 2, 3])
            np.testing.assert_array_equal(handle[f"d{index}"].attrs["common"], small)
            np.testing.assert_array_equal(
                handle[f"large{index}"].attrs["large_common"], large
            )
        for index in range(4):
            np.testing.assert_array_equal(handle[f"fill{index}"][:], [-7] * 4)
        for index in range(6):
            np.testing.assert_array_equal(
                handle[f"component{index}"].attrs[f"unique{index}"], small
            )
        for name in ("dense0", "dense1"):
            np.testing.assert_array_equal(
                handle[name].attrs["common"], dense_common
            )
            for index in range(12):
                np.testing.assert_array_equal(
                    handle[name].attrs[f"{name}_unique{index:02d}"], large + index
                )
        np.testing.assert_array_equal(handle["committed"].attrs["named"], small)
        np.testing.assert_array_equal(handle["committed_value"][:], [4, 5, 6])
        np.testing.assert_array_equal(
            handle["chunked"][:], np.arange(20).reshape(4, 5)
        )

        addresses = {
            name: h5py.h5o.get_info(handle[name].id).addr
            for name in (
                "d0",
                "large0",
                "dense0",
                "dense1",
                "fill0",
                "fill1",
                "fill2",
                "fill3",
            )
        }

    data = path.read_bytes()
    assert data[8] == 2 and data[9] == 8 and data[10] == 8
    assert data.count(b"SMTB") == 1
    assert data.count(b"FRHP") >= 3

    small_messages = object_header_messages(data, addresses["d0"])
    small_attributes = [payload for kind, _, payload in small_messages if kind == 0x0C]
    assert len(small_attributes) == 1
    assert small_attributes[0][0] >= 2
    assert small_attributes[0][1] & 0x03 == 0x03, "datatype/dataspace are not shared"

    large_messages = object_header_messages(data, addresses["large0"])
    shared_attributes = [
        payload
        for kind, flags, payload in large_messages
        if kind == 0x0C and flags & SHARED_FLAG
    ]
    assert shared_attributes, "large attribute is not stored in a SOHM heap"
    assert shared_attributes[0][0] == 3 and shared_attributes[0][1] == 1

    fill_shared = False
    for name in ("fill0", "fill1", "fill2", "fill3"):
        for kind, flags, payload in object_header_messages(data, addresses[name]):
            if kind == 0x05 and flags & SHARED_FLAG:
                assert payload[0] == 3 and payload[1] == 1
                fill_shared = True
    assert fill_shared, "custom fill value is not stored in a SOHM heap"

    for name in ("dense0", "dense1"):
        records = dense_attribute_records(data, addresses[name])
        assert len(records) == 13
        assert all(record[8] & SHARED_FLAG for record in records), (
            "dense attribute B-tree record is not marked shared"
        )

    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_shared_messages_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
