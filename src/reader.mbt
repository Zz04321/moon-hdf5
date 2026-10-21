///|
/// Errors raised by HDF5 reading and writing operations.
pub(all) suberror Hdf5Error {
  UnexpectedEof(Int64)
  InvalidSignature
  UnsupportedSuperblockVersion(Int32)
  UnsupportedAddressSize(Int32)
  UnsupportedFeature(String)
  InvalidArgument(String)
  InvalidField(String)
} derive(Eq, @debug.Debug)

///|
pub extend Hdf5Error with Eq::{not_equal, equal}

///|
pub extend Hdf5Error with @debug.Debug::{to_repr}

///|
pub fn Hdf5Error::message(self : Hdf5Error) -> String {
  match self {
    UnexpectedEof(offset) =>
      "unexpected end of file at byte " + offset.to_string()
    InvalidSignature => "not an HDF5 file"
    UnsupportedSuperblockVersion(version) =>
      "unsupported superblock version " + version.to_string()
    UnsupportedAddressSize(size) =>
      "unsupported address size " + size.to_string()
    UnsupportedFeature(feature) => "unsupported feature: " + feature
    InvalidArgument(reason) => "invalid argument: " + reason
    InvalidField(reason) => "invalid HDF5 data: " + reason
  }
}

///|
/// A bounds-checked cursor over an immutable byte buffer.
priv struct Reader {
  bytes : Bytes
  mut pos : Int32
  limit : Int32
}

///|
fn Reader::new(bytes : Bytes) -> Reader {
  { bytes, pos: 0, limit: bytes.length(), }
}

///|
fn Reader::at(bytes : Bytes, offset : Int32) -> Reader raise Hdf5Error {
  guard offset >= 0 && offset <= bytes.length() else {
    raise InvalidField("offset outside input")
  }
  { bytes, pos: offset, limit: bytes.length(), }
}

///|
fn Reader::position(self : Reader) -> Int32 {
  self.pos
}

///|
fn Reader::remaining(self : Reader) -> Int32 {
  self.limit - self.pos
}

///|
fn Reader::sub(
  self : Reader,
  offset : Int32,
  length : Int32,
) -> Reader raise Hdf5Error {
  guard offset >= 0 &&
    length >= 0 &&
    offset.to_int64() + length.to_int64() <= self.limit.to_int64() else {
    raise InvalidField("sub-reader outside input")
  }
  { bytes: self.bytes, pos: offset, limit: offset + length, }
}

///|
fn Reader::require(self : Reader, n : Int32) -> Unit raise Hdf5Error {
  let end = self.pos.to_int64() + n.to_int64()
  guard n >= 0 && end <= self.limit.to_int64() else {
    raise UnexpectedEof(self.pos.to_int64())
  }
}

///|
fn Reader::skip(self : Reader, n : Int32) -> Unit raise Hdf5Error {
  self.require(n)
  self.pos = self.pos + n
}

///|
fn Reader::read_u8(self : Reader) -> Byte raise Hdf5Error {
  self.require(1)
  let value = self.bytes.unsafe_get(self.pos)
  self.pos = self.pos + 1
  value
}

///|
fn Reader::read_u16_le(self : Reader) -> UInt16 raise Hdf5Error {
  self.require(2)
  let value = self.bytes.unsafe_read_uint16_le(self.pos)
  self.pos = self.pos + 2
  value
}

///|
fn Reader::read_u32_le(self : Reader) -> UInt32 raise Hdf5Error {
  self.require(4)
  let value = self.bytes.unsafe_read_uint32_le(self.pos)
  self.pos = self.pos + 4
  value
}

///|
fn Reader::read_u64_le(self : Reader) -> UInt64 raise Hdf5Error {
  self.require(8)
  let value = self.bytes.unsafe_read_uint64_le(self.pos)
  self.pos = self.pos + 8
  value
}

///|
/// Reads an unsigned little-endian integer of 1, 2, 4 or 8 bytes.
fn Reader::read_uint_le(self : Reader, size : Int32) -> UInt64 raise Hdf5Error {
  match size {
    1 => self.read_u8().to_uint64()
    2 => self.read_u16_le().to_uint64()
    4 => self.read_u32_le().to_uint64()
    8 => self.read_u64_le()
    _ => raise UnsupportedAddressSize(size)
  }
}

///|
fn Reader::read_var_uint_le(
  self : Reader,
  width : Int32,
) -> UInt64 raise Hdf5Error {
  guard width >= 1 && width <= 8 else { raise UnsupportedAddressSize(width) }
  let mut value = 0UL
  for byte_index in 0..<width {
    value = value | (self.read_u8().to_uint64() << (8 * byte_index))
  }
  value
}

///|
fn Reader::read_bytes(self : Reader, n : Int32) -> Bytes raise Hdf5Error {
  self.require(n)
  let view = self.bytes.get_view(start=self.pos, end=self.pos + n)
  let result = match view {
    Some(view) => view.to_owned()
    None => raise UnexpectedEof(self.pos.to_int64())
  }
  self.pos = self.pos + n
  result
}
