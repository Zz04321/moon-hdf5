///|
/// Object header message type for the shared-message ("SOHM") table stored in
/// the superblock extension.
let msg_shared_message_table : Int32 = 0x000F

///|
let shared_message_flag : Int32 = 0x02

///|
let shared_message_heap_id_size : Int32 = 8

///|
priv struct SharedMessageIndex {
  message_types : Int32
  heap_address : UInt64
}

///|
priv struct SharedMessageContext {
  bytes : Bytes
  source : &Hdf5Source?
  offset_size : Int32
  length_size : Int32
  base_address : UInt64
  indexes : Array[SharedMessageIndex]
}

///|
fn shared_message_type_flag(msg_type : Int32) -> Int32? {
  if msg_type == msg_dataspace {
    Some(1 << 1)
  } else if msg_type == msg_datatype {
    Some(1 << 3)
  } else if msg_type == msg_fill_value_old || msg_type == msg_fill_value {
    Some(1 << 5)
  } else if msg_type == msg_filter_pipeline {
    Some(1 << 0x0B)
  } else if msg_type == msg_attribute {
    Some(1 << 0x0C)
  } else {
    None
  }
}

///|
fn shared_message_read_checksummed(
  bytes : Bytes,
  source : &Hdf5Source?,
  address : UInt64,
  content_length : Int32,
  offset_size : Int32,
  base_address : UInt64,
  label : String,
) -> Bytes raise Hdf5Error {
  guard address != heap_undefined_address(offset_size) &&
    content_length >= 0 &&
    content_length <= 0x7FFFFFFF - metadata_checksum_size else {
    raise InvalidField(label + " has an invalid address or size")
  }
  match source {
    Some(random_access) => {
      let offset = source_address_to_offset(address, base_address)
      source_read_checksummed(random_access, offset, content_length, label)
    }
    None => {
      let offset = address_to_offset(address, base_address)
      let reader = Reader::at(bytes, offset)
      let data = reader.read_bytes(content_length + metadata_checksum_size)
      verify_metadata_checksum(data, 0, content_length, label)
      data
    }
  }
}

///|
fn parse_shared_message_context(
  bytes : Bytes,
  source : &Hdf5Source?,
  superblock : Superblock,
) -> SharedMessageContext raise Hdf5Error {
  let indexes : Array[SharedMessageIndex] = []
  let context : SharedMessageContext = {
    bytes,
    source,
    offset_size: superblock.offset_size,
    length_size: superblock.length_size,
    base_address: superblock.base_address,
    indexes,
  }
  if superblock.version < 2 ||
    superblock.superblock_extension_address ==
    heap_undefined_address(superblock.offset_size) {
    return context
  }
  let extension = match source {
    Some(random_access) =>
      parse_source_object_header(
        random_access,
        superblock.superblock_extension_address,
        superblock.offset_size,
        superblock.length_size,
        superblock.base_address,
      )
    None =>
      parse_object_header(
        bytes,
        superblock.superblock_extension_address,
        superblock.offset_size,
        superblock.length_size,
        superblock.base_address,
      )
  }
  let message = match extension.first(msg_shared_message_table) {
    Some(message) => message
    None => return context
  }
  let reader = Reader::new(message.data)
  guard reader.read_u8() == b'\x00' else {
    raise UnsupportedFeature("unsupported shared-message table version")
  }
  let table_address = reader.read_uint_le(superblock.offset_size)
  let index_count = reader.read_u8().to_int()
  // HDF5 reserves six bytes after the index count. The message may also be
  // padded by the object-header encoder, so only the defined prefix is read.
  guard table_address != heap_undefined_address(superblock.offset_size) &&
    index_count >= 1 &&
    index_count <= 8 else {
    raise InvalidField("invalid shared-message table address or index count")
  }
  let index_size = 14 + superblock.offset_size * 2
  let table_size = 4 + index_count * index_size
  guard table_size <= 0x7FFFFFFF - metadata_checksum_size else {
    raise InvalidField("shared-message table is too large")
  }
  let table = shared_message_read_checksummed(
    bytes,
    source,
    table_address,
    table_size,
    superblock.offset_size,
    superblock.base_address,
    "shared-message table",
  )
  let table_reader = Reader::new(table)
  guard table_reader.read_bytes(4) == b"SMTB" else {
    raise InvalidField("invalid shared-message table signature")
  }
  for _ in 0..<index_count {
    guard table_reader.read_u8() == b'\x00' else {
      raise UnsupportedFeature("unsupported shared-message index version")
    }
    let index_type = table_reader.read_u8().to_int()
    guard index_type == 0 || index_type == 1 else {
      raise InvalidField("invalid shared-message index type")
    }
    let message_types = table_reader.read_u16_le().to_int()
    let _minimum_size = table_reader.read_u32_le()
    let _list_maximum = table_reader.read_u16_le()
    let _btree_minimum = table_reader.read_u16_le()
    let _message_count = table_reader.read_u16_le()
    let _index_address = table_reader.read_uint_le(superblock.offset_size)
    let heap_address = table_reader.read_uint_le(superblock.offset_size)
    guard heap_address != heap_undefined_address(superblock.offset_size) else {
      raise InvalidField("shared-message index has no fractal heap")
    }
    indexes.push({ message_types, heap_address, })
  }
  guard table_reader.remaining() == metadata_checksum_size else {
    raise InvalidField("shared-message table size mismatch")
  }
  context
}

///|
fn SharedMessageContext::heap_address(
  self : SharedMessageContext,
  msg_type : Int32,
) -> UInt64? {
  let flag = shared_message_type_flag(msg_type)
  match flag {
    Some(flag) => {
      for index in self.indexes {
        if (index.message_types & flag) != 0 {
          return Some(index.heap_address)
        }
      }
      None
    }
    None => None
  }
}

///|
/// Reads a message stored directly in a SOHM fractal heap. Dense attribute
/// B-tree records use this form when their object-header shared flag is set.
fn SharedMessageContext::resolve_sohm_heap_id(
  self : SharedMessageContext,
  heap_id : Bytes,
  msg_type : Int32,
) -> Bytes raise Hdf5Error {
  guard heap_id.length() == shared_message_heap_id_size else {
    raise InvalidField("shared-message heap ID has an invalid size")
  }
  let heap_address = match self.heap_address(msg_type) {
    Some(address) => address
    None =>
      raise UnsupportedFeature("shared message has no matching SOHM index")
  }
  let heap = match self.source {
    Some(source) =>
      parse_source_fractal_heap(
        source,
        heap_address,
        self.offset_size,
        self.length_size,
        self.base_address,
      )
    None =>
      parse_fractal_heap(
        self.bytes,
        heap_address,
        self.offset_size,
        self.length_size,
        self.base_address,
      )
  }
  heap.get_data(heap_id)
}

///|
fn SharedMessageContext::object_header_at(
  self : SharedMessageContext,
  address : UInt64,
) -> ObjectHeader raise Hdf5Error {
  match self.source {
    Some(source) =>
      parse_source_object_header(
        source,
        address,
        self.offset_size,
        self.length_size,
        self.base_address,
      )
    None =>
      parse_object_header(
        self.bytes,
        address,
        self.offset_size,
        self.length_size,
        self.base_address,
      )
  }
}

///|
fn SharedMessageContext::resolve_shared_message(
  self : SharedMessageContext,
  payload : Bytes,
  msg_type : Int32,
  visited : Array[UInt64],
) -> Bytes raise Hdf5Error {
  let reader = Reader::new(payload)
  let version = reader.read_u8().to_int()
  guard version >= 1 && version <= 3 else {
    raise UnsupportedFeature("unsupported shared-message encoding version")
  }
  let encoded_type = reader.read_u8().to_int()
  let shared_type = if version == 1 { 2 } else { encoded_type }
  if shared_type == 1 {
    guard version == 3 else {
      raise InvalidField("shared-message heap ID requires version 3")
    }
    let heap_id = reader.read_bytes(shared_message_heap_id_size)
    guard reader.remaining() == 0 else {
      raise InvalidField("shared-message heap ID has trailing bytes")
    }
    return self.resolve_sohm_heap_id(heap_id, msg_type)
  }
  let address = if version == 1 {
    reader.skip(6)
    let _local_heap_address = reader.read_uint_le(self.length_size)
    reader.read_uint_le(self.offset_size)
  } else {
    guard shared_type == 2 else {
      if shared_type == 3 {
        raise InvalidField("a stored shared message cannot be stored here")
      }
      raise InvalidField("invalid shared-message storage type")
    }
    reader.read_uint_le(self.offset_size)
  }
  guard reader.remaining() == 0 else {
    raise InvalidField("shared-message address has trailing bytes")
  }
  guard address != heap_undefined_address(self.offset_size) else {
    raise InvalidField("shared-message object header address is undefined")
  }
  guard !visited.contains(address) else {
    raise InvalidField("shared-message object header cycle")
  }
  let header = self.object_header_at(address)
  let target = match header.first(msg_type) {
    Some(message) => message
    None =>
      raise InvalidField(
        "shared-message target has no matching object header message",
      )
  }
  let next_visited = visited.copy()
  next_visited.push(address)
  if (target.flags.to_int() & shared_message_flag) != 0 {
    self.resolve_shared_message(target.data, msg_type, next_visited)
  } else {
    target.data
  }
}

///|
fn SharedMessageContext::resolve_object_header(
  self : SharedMessageContext,
  header : ObjectHeader,
  address : UInt64,
) -> ObjectHeader raise Hdf5Error {
  let messages : Array[HeaderMessage] = []
  let visited : Array[UInt64] = [address]
  for message in header.messages {
    if (message.flags.to_int() & shared_message_flag) != 0 {
      let data = self.resolve_shared_message(
        message.data,
        message.msg_type,
        visited,
      )
      messages.push({
        msg_type: message.msg_type,
        flags: (message.flags.to_int() & (0xFF ^ shared_message_flag)).to_byte(),
        data,
      })
    } else {
      messages.push(message)
    }
  }
  { _version: header._version, messages, }
}
