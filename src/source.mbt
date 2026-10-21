///|
/// A random-access byte source for an HDF5 file.
///
/// Implementations must return exactly `length` bytes or raise an error.
pub(open) trait Hdf5Source: @debug.Debug {
  fn length(Self) -> UInt64
  fn read_at(Self, offset : UInt64, length : Int32) -> Bytes raise Hdf5Error
}

///|
fn source_address_to_offset(
  address : UInt64,
  base_address : UInt64,
) -> UInt64 raise Hdf5Error {
  guard address != 0xFFFFFFFFFFFFFFFFUL &&
    base_address <= 0xFFFFFFFFFFFFFFFFUL - address else {
    raise InvalidField("HDF5 address is undefined or out of range")
  }
  base_address + address
}

///|
fn source_read_available(
  source : &Hdf5Source,
  offset : UInt64,
  maximum : Int32,
) -> Bytes raise Hdf5Error {
  let length = source.length()
  guard offset <= length else {
    raise UnexpectedEof(offset.reinterpret_as_int64())
  }
  let remaining = length - offset
  let available = if remaining < maximum.to_uint64() {
    remaining.to_int()
  } else {
    maximum
  }
  source.read_at(offset, available)
}

///|
fn source_read_checksummed(
  source : &Hdf5Source,
  offset : UInt64,
  content_length : Int32,
  description : String,
) -> Bytes raise Hdf5Error {
  guard content_length >= 0 && content_length <= 0x7FFFFFFF - 4 else {
    raise InvalidField(description + " is too large")
  }
  let bytes = source.read_at(offset, content_length + 4)
  verify_metadata_checksum(bytes, 0, content_length, description)
  bytes
}

///|
fn find_source_superblock_offset(
  source : &Hdf5Source,
) -> UInt64 raise Hdf5Error {
  let length = source.length()
  let mut offset = 0UL
  while offset <= 1048576UL && offset + 8UL <= length {
    if source.read_at(offset, 8) == hdf5_signature {
      return offset
    }
    offset = if offset == 0UL { 512UL } else { offset * 2UL }
  }
  raise InvalidSignature
}

///|
fn parse_source_superblock(source : &Hdf5Source) -> Superblock raise Hdf5Error {
  let offset = find_source_superblock_offset(source)
  guard offset <= 0x7FFFFFFFUL else {
    raise InvalidField("superblock offset is out of range")
  }
  let bytes = source_read_available(source, offset, 96)
  let parsed = parse_superblock(bytes)
  {
    superblock_offset: offset.to_int(),
    superblock_end: offset.to_int() + parsed.superblock_end,
    version: parsed.version,
    offset_size: parsed.offset_size,
    length_size: parsed.length_size,
    file_consistency_flags: parsed.file_consistency_flags,
    base_address: parsed.base_address,
    free_space_address: parsed.free_space_address,
    end_of_file_address: parsed.end_of_file_address,
    driver_info_address: parsed.driver_info_address,
    superblock_extension_address: parsed.superblock_extension_address,
    root_group_address: parsed.root_group_address,
    root_group_object_header: parsed.root_group_object_header,
    group_leaf_node_k: parsed.group_leaf_node_k,
    group_internal_node_k: parsed.group_internal_node_k,
    superblock_checksum: parsed.superblock_checksum,
  }
}

///|
fn parse_source_object_header(
  source : &Hdf5Source,
  address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> ObjectHeader raise Hdf5Error {
  let absolute = source_address_to_offset(address, base_address)
  let prefix = source_read_available(source, absolute, 32)
  let reader = Reader::new(prefix)
  let first = reader.read_u8()
  match first {
    b'\x01' => {
      guard prefix.length() >= 16 else {
        raise UnexpectedEof(absolute.reinterpret_as_int64())
      }
      let header_size = Reader::at(prefix, 8).read_u32_le().reinterpret_as_int()
      guard header_size >= 0 && header_size <= 0x7FFFFFFF - 16 else {
        raise InvalidField("object header chunk size is too large")
      }
      let bytes = source.read_at(absolute, 16 + header_size)
      let header = Reader::new(bytes)
      guard header.read_u8() == b'\x01' else {
        raise InvalidField("expected object header version 1")
      }
      header.skip(7)
      let size = header.read_u32_le().reinterpret_as_int()
      header.skip(4)
      let messages : Array[HeaderMessage] = []
      parse_source_v1_messages(
        source,
        header.sub(16, size),
        offset_size,
        length_size,
        base_address,
        [absolute],
        messages,
      )
      { version: 1, messages, }
    }
    b'O' => {
      guard reader.read_bytes(3) == b"HDR" else {
        raise InvalidField("expected OHDR signature")
      }
      guard reader.read_u8() == b'\x02' else {
        raise InvalidField("expected object header version 2")
      }
      let flags = reader.read_u8().to_int()
      if (flags & 0x20) != 0 {
        reader.skip(16)
      }
      if (flags & 0x10) != 0 {
        reader.skip(4)
      }
      let chunk_size = reader.read_uint_le(1 << (flags & 0x03))
      guard chunk_size <= 0x7FFFFFFFUL - reader.position().to_uint64() - 4UL else {
        raise InvalidField("object header chunk size is too large")
      }
      let payload_start = reader.position()
      let total = payload_start + chunk_size.to_int() + 4
      let bytes = source.read_at(absolute, total)
      verify_metadata_checksum(
        bytes,
        0,
        payload_start + chunk_size.to_int(),
        "object header",
      )
      let messages : Array[HeaderMessage] = []
      parse_source_v2_messages(
        source,
        Reader::new(bytes).sub(payload_start, chunk_size.to_int()),
        offset_size,
        length_size,
        base_address,
        if (flags & 0x04) != 0 {
          2
        } else {
          0
        },
        [absolute],
        messages,
      )
      { version: 2, messages, }
    }
    _ => raise InvalidField("unknown object header version")
  }
}

///|
fn parse_source_v1_messages(
  source : &Hdf5Source,
  block : Reader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  visited : Array[UInt64],
  messages : Array[HeaderMessage],
) -> Unit raise Hdf5Error {
  while block.remaining() >= 8 {
    let msg_type = block.read_u16_le().to_int()
    let size = block.read_u16_le().to_int()
    let flags = block.read_u8()
    block.skip(3)
    let data_offset = block.position()
    let data = block.read_bytes(size)
    messages.push({ msg_type, flags, data_offset, data, })
    if msg_type == msg_object_continuation {
      let continuation = Reader::new(data)
      let continuation_address = continuation.read_uint_le(offset_size)
      let continuation_length = continuation.read_uint_le(length_size)
      let continuation_offset = source_address_to_offset(
        continuation_address, base_address,
      )
      if !visited.contains(continuation_offset) {
        guard continuation_length <= 0x7FFFFFFFUL else {
          raise InvalidField("object header continuation is too large")
        }
        visited.push(continuation_offset)
        parse_source_v1_messages(
          source,
          Reader::new(
            source.read_at(continuation_offset, continuation_length.to_int()),
          ),
          offset_size,
          length_size,
          base_address,
          visited,
          messages,
        )
      }
    }
  }
}

///|
fn parse_source_v2_messages(
  source : &Hdf5Source,
  block : Reader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  creation_order_size : Int32,
  visited : Array[UInt64],
  messages : Array[HeaderMessage],
) -> Unit raise Hdf5Error {
  let prefix_size = 4 + creation_order_size
  while block.remaining() >= prefix_size {
    let msg_type = block.read_u8().to_int()
    let size = block.read_u16_le().to_int()
    let flags = block.read_u8()
    if creation_order_size == 2 {
      block.skip(2)
    }
    let data_offset = block.position()
    let data = block.read_bytes(size)
    messages.push({ msg_type, flags, data_offset, data, })
    if msg_type == msg_object_continuation {
      let continuation = Reader::new(data)
      let continuation_address = continuation.read_uint_le(offset_size)
      let continuation_length = continuation.read_uint_le(length_size)
      let continuation_offset = source_address_to_offset(
        continuation_address, base_address,
      )
      if !visited.contains(continuation_offset) {
        guard continuation_length >= 8UL && continuation_length <= 0x7FFFFFFFUL else {
          raise InvalidField("invalid object header continuation length")
        }
        visited.push(continuation_offset)
        let bytes = source.read_at(
          continuation_offset,
          continuation_length.to_int(),
        )
        guard bytes.get_view(start=0, end=4).unwrap().equal_to_bytes(b"OCHK") else {
          raise InvalidField("object header continuation is missing OCHK")
        }
        verify_metadata_checksum(
          bytes,
          0,
          bytes.length() - metadata_checksum_size,
          "object header continuation",
        )
        parse_source_v2_messages(
          source,
          Reader::new(bytes).sub(4, bytes.length() - 8),
          offset_size,
          length_size,
          base_address,
          creation_order_size,
          visited,
          messages,
        )
      }
    }
  }
}

///|
priv struct SourceLocalHeap {
  data : Bytes
}

///|
fn SourceLocalHeap::get_string(
  self : SourceLocalHeap,
  offset : UInt64,
) -> String raise Hdf5Error {
  guard offset < self.data.length().to_uint64() else {
    raise InvalidField("local heap string offset outside data segment")
  }
  let start = offset.to_int()
  let mut end = start
  while end < self.data.length() && self.data.unsafe_get(end) != b'\x00' {
    end = end + 1
  }
  guard end < self.data.length() else {
    raise InvalidField("unterminated local heap string")
  }
  @utf8.decode(self.data.get_view(start~, end~).unwrap()) catch {
    _ => raise InvalidField("invalid UTF-8 string in local heap")
  }
}

///|
fn read_source_local_heap(
  source : &Hdf5Source,
  address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> SourceLocalHeap raise Hdf5Error {
  let absolute = source_address_to_offset(address, base_address)
  let header_size = 8 + length_size * 2 + offset_size
  let reader = Reader::new(source.read_at(absolute, header_size))
  guard reader.read_bytes(4) == b"HEAP" && reader.read_u8() == b'\x00' else {
    raise InvalidField("expected HEAP signature or version 0")
  }
  reader.skip(3)
  let data_size = reader.read_uint_le(length_size)
  guard data_size <= 0x7FFFFFFFUL else {
    raise InvalidField("local heap size is too large")
  }
  reader.read_uint_le(length_size) |> ignore
  let data_address = source_address_to_offset(
    reader.read_uint_le(offset_size),
    base_address,
  )
  { data: source.read_at(data_address, data_size.to_int()), }
}

///|
fn read_source_symbol_node(
  source : &Hdf5Source,
  address : UInt64,
  offset_size : Int32,
  base_address : UInt64,
  heap : SourceLocalHeap,
  links : Array[GroupLink],
) -> Unit raise Hdf5Error {
  let absolute = source_address_to_offset(address, base_address)
  let prefix = source.read_at(absolute, 8)
  let count_reader = Reader::new(prefix)
  guard count_reader.read_bytes(4) == b"SNOD" &&
    count_reader.read_u8() == b'\x01' else {
    raise InvalidField("expected SNOD signature or version 1")
  }
  count_reader.skip(1)
  let symbol_count = count_reader.read_u16_le().to_int()
  let entry_size = offset_size * 2 + 24
  let reader = Reader::new(
    source.read_at(absolute, 8 + symbol_count * entry_size),
  )
  reader.skip(8)
  for _ in 0..<symbol_count {
    let name_offset = reader.read_uint_le(offset_size)
    let object_header_address = reader.read_uint_le(offset_size)
    let cache_type = reader.read_u32_le()
    reader.skip(4)
    let scratch = reader.read_bytes(16)
    let link_target = if cache_type == 2U {
      Some(heap.get_string(Reader::new(scratch).read_u32_le().to_uint64()))
    } else {
      None
    }
    links.push({
      name: heap.get_string(name_offset),
      object_header_address,
      cache_type,
      link_target,
    })
  }
}

///|
fn read_source_group_btree(
  source : &Hdf5Source,
  address : UInt64,
  expected_level : Int32,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  heap : SourceLocalHeap,
  links : Array[GroupLink],
  visited : Array[UInt64],
) -> Unit raise Hdf5Error {
  let absolute = source_address_to_offset(address, base_address)
  guard !visited.contains(absolute) else { return }
  visited.push(absolute)
  let prefix_size = 8 + offset_size * 2
  let prefix = Reader::new(source.read_at(absolute, prefix_size))
  guard prefix.read_bytes(4) == b"TREE" && prefix.read_u8() == b'\x00' else {
    raise InvalidField("expected group TREE signature")
  }
  let level = prefix.read_u8().to_int()
  guard expected_level < 0 || level == expected_level else {
    raise InvalidField("unexpected B-tree level")
  }
  let entries = prefix.read_u16_le().to_int()
  let total = prefix_size + entries * (length_size + offset_size) + length_size
  let reader = Reader::new(source.read_at(absolute, total))
  reader.skip(prefix_size)
  for _ in 0..<entries {
    reader.skip(length_size)
    let child = reader.read_uint_le(offset_size)
    if level > 0 {
      read_source_group_btree(
        source,
        child,
        level - 1,
        offset_size,
        length_size,
        base_address,
        heap,
        links,
        visited,
      )
    } else {
      read_source_symbol_node(
        source, child, offset_size, base_address, heap, links,
      )
    }
  }
}

///|
fn read_source_group_links_v1(
  source : &Hdf5Source,
  btree_address : UInt64,
  heap_address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Array[GroupLink] raise Hdf5Error {
  let heap = read_source_local_heap(
    source, heap_address, offset_size, length_size, base_address,
  )
  let links : Array[GroupLink] = []
  read_source_group_btree(
    source,
    btree_address,
    -1,
    offset_size,
    length_size,
    base_address,
    heap,
    links,
    [],
  )
  links
}

///|
pub impl Hdf5Source for Bytes with fn length(self : Bytes) -> UInt64 {
  self.length().to_uint64()
}

///|
pub impl Hdf5Source for Bytes with fn read_at(
  self : Bytes,
  offset : UInt64,
  length : Int32,
) -> Bytes raise Hdf5Error {
  let total = self.length().to_uint64()
  guard length >= 0 && offset <= total && length.to_uint64() <= total - offset else {
    raise UnexpectedEof(offset.reinterpret_as_int64())
  }
  self
  .get_view(start=offset.to_int(), end=offset.to_int() + length)
  .unwrap()
  .to_owned()
}
