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
