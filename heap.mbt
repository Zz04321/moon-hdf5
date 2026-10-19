///|
/// A parsed HDF5 local heap.
pub(all) struct LocalHeap {
  bytes : Bytes
  data_address : Int32
  data_size : Int32
  offset_to_free_list : UInt64
} derive(@debug.Debug)

///|
pub extend LocalHeap with @debug.Debug::{to_repr}

///|
pub fn parse_local_heap(
  bytes : Bytes,
  address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> LocalHeap raise Hdf5Error {
  let absolute = address_to_offset(address, base_address)
  let reader = Reader::at(bytes, absolute)
  let signature = reader.read_bytes(4)
  guard signature == b"HEAP" else {
    raise InvalidField("expected HEAP signature")
  }
  let version = reader.read_u8().to_int()
  guard version == 0 else {
    raise InvalidField("unsupported local heap version")
  }
  reader.skip(3)
  let data_size = reader.read_uint_le(length_size).to_int()
  let offset_to_free_list = reader.read_uint_le(length_size)
  let data_address = address_to_offset(
    reader.read_uint_le(offset_size),
    base_address,
  )
  guard data_size >= 0 else { raise InvalidField("negative local heap size") }
  { bytes, data_address, data_size, offset_to_free_list, }
}

///|
/// Reads a null-terminated UTF-8 string from the local heap data segment.
pub fn LocalHeap::get_string(
  self : LocalHeap,
  offset : UInt64,
) -> String raise Hdf5Error {
  guard offset <= 0x7FFFFFFFUL else {
    raise InvalidField("local heap offset out of range")
  }
  let start = self.data_address + offset.to_int()
  let end = self.data_address + self.data_size
  guard start >= self.data_address && start < end else {
    raise InvalidField("local heap string offset outside data segment")
  }
  let mut cursor = start
  while cursor < end && self.bytes.unsafe_get(cursor) != b'\x00' {
    cursor = cursor + 1
  }
  guard cursor < end else {
    raise InvalidField("unterminated local heap string")
  }
  let view = self.bytes.get_view(start~, end=cursor).unwrap()
  @utf8.decode(view) catch {
    _ => raise InvalidField("invalid UTF-8 string in local heap")
  }
}
