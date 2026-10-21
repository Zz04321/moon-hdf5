///|
/// The two addresses stored in a symbol table message.
priv struct SymbolTableInfo {
  btree_address : UInt64
  heap_address : UInt64
}

///|
fn parse_symbol_table_message(
  message : HeaderMessage,
  offset_size : Int32,
  _length_size : Int32,
) -> SymbolTableInfo raise Hdf5Error {
  guard message.msg_type == msg_symbol_table else {
    raise InvalidField("message is not a symbol table message")
  }
  let reader = Reader::new(message.data)
  let btree_address = reader.read_uint_le(offset_size)
  let heap_address = reader.read_uint_le(offset_size)
  { btree_address, heap_address, }
}

///|
/// A hard or soft link found in a version 1 group.
priv struct GroupLink {
  name : String
  object_header_address : UInt64
  cache_type : UInt32
  link_target : String?
}

///|
fn read_symbol_table_node(
  bytes : Bytes,
  address : UInt64,
  offset_size : Int32,
  _length_size : Int32,
  base_address : UInt64,
  heap : LocalHeap,
  links : Array[GroupLink],
) -> Unit raise Hdf5Error {
  let absolute = address_to_offset(address, base_address)
  let reader = Reader::at(bytes, absolute)
  let signature = reader.read_bytes(4)
  guard signature == b"SNOD" else {
    raise InvalidField("expected SNOD signature")
  }
  let version = reader.read_u8().to_int()
  guard version == 1 else {
    raise InvalidField("unsupported symbol table node version")
  }
  reader.skip(1)
  let symbol_count = reader.read_u16_le().to_int()
  for _ in 0..<symbol_count {
    let name_offset = reader.read_uint_le(offset_size)
    let object_header_address = reader.read_uint_le(offset_size)
    let cache_type = reader.read_u32_le()
    let _reserved = reader.read_u32_le()
    let scratch = reader.read_bytes(16)
    let name = heap.get_string(name_offset)
    let link_target = if cache_type == 2U {
      let link_offset = Reader::new(scratch).read_u32_le().to_uint64()
      Some(heap.get_string(link_offset))
    } else {
      None
    }
    links.push({ name, object_header_address, cache_type, link_target, })
  }
}

///|
fn read_group_btree_node(
  bytes : Bytes,
  address : UInt64,
  expected_level : Int32,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  heap : LocalHeap,
  links : Array[GroupLink],
  visited : Array[Int32],
) -> Unit raise Hdf5Error {
  let absolute = address_to_offset(address, base_address)
  guard !visited.contains(absolute) else { return }
  visited.push(absolute)
  let reader = Reader::at(bytes, absolute)
  let signature = reader.read_bytes(4)
  guard signature == b"TREE" else {
    raise InvalidField("expected TREE signature")
  }
  let node_type = reader.read_u8().to_int()
  guard node_type == 0 else {
    raise InvalidField("B-tree node is not a group node")
  }
  let node_level = reader.read_u8().to_int()
  if expected_level >= 0 {
    guard node_level == expected_level else {
      raise InvalidField("unexpected B-tree level")
    }
  }
  let entries_used = reader.read_u16_le().to_int()
  let _left_sibling = reader.read_uint_le(offset_size)
  let _right_sibling = reader.read_uint_le(offset_size)
  for _ in 0..<entries_used {
    let _key = reader.read_uint_le(length_size)
    let child = reader.read_uint_le(offset_size)
    if node_level > 0 {
      read_group_btree_node(
        bytes,
        child,
        node_level - 1,
        offset_size,
        length_size,
        base_address,
        heap,
        links,
        visited,
      )
    } else {
      read_symbol_table_node(
        bytes, child, offset_size, length_size, base_address, heap, links,
      )
    }
  }
  let _final_key = reader.read_uint_le(length_size)
}

///|
/// Reads all links from a version 1 group (symbol table message).
fn read_group_links_v1(
  bytes : Bytes,
  btree_address : UInt64,
  heap_address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Array[GroupLink] raise Hdf5Error {
  let heap = parse_local_heap(
    bytes, heap_address, offset_size, length_size, base_address,
  )
  let links : Array[GroupLink] = []
  let visited : Array[Int32] = []
  read_group_btree_node(
    bytes, btree_address, -1, offset_size, length_size, base_address, heap, links,
    visited,
  )
  links
}
