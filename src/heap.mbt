///|
/// A parsed HDF5 local heap.
priv struct LocalHeap {
  bytes : Bytes
  data_address : Int32
  data_size : Int32
}

///|
fn parse_local_heap(
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
  let stored_size = reader.read_uint_le(length_size)
  guard stored_size <= 0x7FFFFFFFUL else {
    raise InvalidField("local heap size is too large")
  }
  let data_size = stored_size.to_int()
  let _offset_to_free_list = reader.read_uint_le(length_size)
  let data_address = address_to_offset(
    reader.read_uint_le(offset_size),
    base_address,
  )
  Reader::new(bytes).sub(data_address, data_size) |> ignore
  { bytes, data_address, data_size, }
}

///|
/// Reads a null-terminated UTF-8 string from the local heap data segment.
fn LocalHeap::get_string(
  self : LocalHeap,
  offset : UInt64,
) -> String raise Hdf5Error {
  guard self.data_address >= 0 &&
    self.data_address <= self.bytes.length() &&
    self.data_size >= 0 &&
    self.data_size <= self.bytes.length() - self.data_address else {
    raise InvalidField("local heap data segment outside input")
  }
  guard offset < self.data_size.to_uint64() else {
    raise InvalidField("local heap string offset outside data segment")
  }
  let start = self.data_address + offset.to_int()
  let end = self.data_address + self.data_size
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
