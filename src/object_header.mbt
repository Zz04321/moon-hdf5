///|
/// HDF5 object header message types.
pub let msg_nil : Int32 = 0x0000

///|
pub let msg_dataspace : Int32 = 0x0001

///|
pub let msg_link_info : Int32 = 0x0002

///|
pub let msg_datatype : Int32 = 0x0003

///|
pub let msg_fill_value_old : Int32 = 0x0004

///|
pub let msg_fill_value : Int32 = 0x0005

///|
pub let msg_link : Int32 = 0x0006

///|
pub let msg_external_data_files : Int32 = 0x0007

///|
pub let msg_data_layout : Int32 = 0x0008

///|
pub let msg_bogus : Int32 = 0x0009

///|
pub let msg_group_info : Int32 = 0x000A

///|
pub let msg_filter_pipeline : Int32 = 0x000B

///|
pub let msg_attribute : Int32 = 0x000C

///|
pub let msg_object_comment : Int32 = 0x000D

///|
pub let msg_shared_message_table : Int32 = 0x000F

///|
pub let msg_object_continuation : Int32 = 0x0010

///|
pub let msg_symbol_table : Int32 = 0x0011

///|
pub let msg_modification_time : Int32 = 0x0012

///|
pub let msg_attribute_info : Int32 = 0x0015

///|
pub let msg_object_reference_count : Int32 = 0x0016

///|
let undefined_address : UInt64 = 0xFFFFFFFFFFFFFFFFUL

///|
fn address_to_offset(
  address : UInt64,
  base_address : UInt64,
) -> Int32 raise Hdf5Error {
  guard address != undefined_address else {
    raise InvalidField("HDF5 address is undefined")
  }
  let absolute = base_address + address
  guard absolute >= address && absolute <= 0x7FFFFFFFUL else {
    raise InvalidField("HDF5 address is undefined or out of range")
  }
  absolute.to_int()
}

///|
/// One message from an object header.
pub(all) struct HeaderMessage {
  msg_type : Int32
  flags : Byte
  data_offset : Int32
  data : Bytes
} derive(@debug.Debug)

///|
pub extend HeaderMessage with @debug.Debug::{to_repr}

///|
/// A parsed object header.
pub(all) struct ObjectHeader {
  version : Int32
  messages : Array[HeaderMessage]
} derive(@debug.Debug)

///|
pub extend ObjectHeader with @debug.Debug::{to_repr}

///|
pub fn ObjectHeader::find(
  self : ObjectHeader,
  msg_type : Int32,
) -> Array[HeaderMessage] {
  self.messages.filter(message => message.msg_type == msg_type)
}

///|
pub fn ObjectHeader::first(
  self : ObjectHeader,
  msg_type : Int32,
) -> HeaderMessage? {
  for message in self.messages {
    if message.msg_type == msg_type {
      return Some(message)
    }
  }
  None
}

///|
fn parse_v1_message_block(
  bytes : Bytes,
  block : Reader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  visited : Array[Int32],
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
      let address = continuation.read_uint_le(offset_size)
      let length = continuation.read_uint_le(length_size)
      let continuation_offset = address_to_offset(address, base_address)
      if !visited.contains(continuation_offset) {
        visited.push(continuation_offset)
        guard length <= 0x7FFFFFFFUL else {
          raise InvalidField("object header continuation is too large")
        }
        let next = Reader::at(bytes, continuation_offset)
        parse_v1_message_block(
          bytes,
          next.sub(continuation_offset, length.to_int()),
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
fn parse_object_header_v1(
  bytes : Bytes,
  address : Int32,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> ObjectHeader raise Hdf5Error {
  let reader = Reader::at(bytes, address)
  let version = reader.read_u8().to_int()
  guard version == 1 else {
    raise InvalidField("expected object header version 1")
  }
  let _reserved = reader.read_u8()
  let _total_messages = reader.read_u16_le()
  let _reference_count = reader.read_u32_le()
  let header_size = reader.read_u32_le().reinterpret_as_int()
  let _padding = reader.read_u32_le()
  guard header_size >= 0 else {
    raise InvalidField("negative object header size")
  }
  let payload_start = reader.position()
  let block = reader.sub(payload_start, header_size)
  let messages : Array[HeaderMessage] = []
  let visited : Array[Int32] = [address]
  parse_v1_message_block(
    bytes, block, offset_size, length_size, base_address, visited, messages,
  )
  { version, messages, }
}

///|
fn parse_v2_message_block(
  bytes : Bytes,
  block : Reader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  creation_order_size : Int32,
  visited : Array[Int32],
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
      let address = continuation.read_uint_le(offset_size)
      let length = continuation.read_uint_le(length_size)
      let continuation_offset = address_to_offset(address, base_address)
      if !visited.contains(continuation_offset) {
        visited.push(continuation_offset)
        guard length >= 8UL && length <= 0x7FFFFFFFUL else {
          raise InvalidField("invalid object header continuation length")
        }
        let next = Reader::at(bytes, continuation_offset)
        let complete = next.sub(continuation_offset, length.to_int())
        let signature = complete.read_bytes(4)
        guard signature == b"OCHK" else {
          raise InvalidField("object header continuation is missing OCHK")
        }
        parse_v2_message_block(
          bytes,
          complete.sub(continuation_offset + 4, length.to_int() - 8),
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
fn parse_object_header_v2(
  bytes : Bytes,
  address : Int32,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> ObjectHeader raise Hdf5Error {
  let reader = Reader::at(bytes, address)
  let signature = reader.read_bytes(4)
  guard signature == b"OHDR" else {
    raise InvalidField("expected OHDR signature")
  }
  let version = reader.read_u8().to_int()
  guard version == 2 else {
    raise InvalidField("expected object header version 2")
  }
  let flags = reader.read_u8().to_int()
  let creation_order_size = if (flags & 0x04) != 0 { 2 } else { 0 }
  if (flags & 0x20) != 0 {
    reader.skip(16)
  }
  if (flags & 0x10) != 0 {
    reader.skip(4)
  }
  let raw_chunk_size = reader.read_uint_le(1 << (flags & 0x03))
  guard raw_chunk_size <= 0x7FFFFFFFUL else {
    raise InvalidField("object header chunk size is too large")
  }
  let chunk_size = raw_chunk_size.to_int()
  let payload_start = reader.position()
  let block = reader.sub(payload_start, chunk_size)
  let _checksum = reader.sub(payload_start + chunk_size, 4)
  let messages : Array[HeaderMessage] = []
  let visited : Array[Int32] = [address]
  parse_v2_message_block(
    bytes, block, offset_size, length_size, base_address, creation_order_size, visited,
    messages,
  )
  { version, messages, }
}

///|
/// Parses an HDF5 object header at `address`.
pub fn parse_object_header(
  bytes : Bytes,
  address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> ObjectHeader raise Hdf5Error {
  let offset = address_to_offset(address, base_address)
  let reader = Reader::at(bytes, offset)
  let first = reader.read_u8()
  match first {
    b'\x01' =>
      parse_object_header_v1(
        bytes, offset, offset_size, length_size, base_address,
      )
    b'O' =>
      parse_object_header_v2(
        bytes, offset, offset_size, length_size, base_address,
      )
    _ => raise InvalidField("unknown object header version")
  }
}
