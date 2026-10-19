///|
/// Errors raised while reading an HDF5 file.
pub(all) suberror Hdf5Error {
  UnexpectedEof(Int64)
  InvalidSignature
  UnsupportedSuperblockVersion(Int32)
  UnsupportedAddressSize(Int32)
  UnsupportedFeature(String)
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
    InvalidField(reason) => "invalid HDF5 data: " + reason
  }
}

///|
/// A bounds-checked cursor over an immutable byte buffer.
pub(all) struct Reader {
  bytes : Bytes
  mut pos : Int32
  limit : Int32
}

///|
pub fn Reader::new(bytes : Bytes) -> Reader {
  { bytes, pos: 0, limit: bytes.length(), }
}

///|
pub fn Reader::at(bytes : Bytes, offset : Int32) -> Reader raise Hdf5Error {
  guard offset >= 0 && offset <= bytes.length() else {
    raise InvalidField("offset outside input")
  }
  { bytes, pos: offset, limit: bytes.length(), }
}

///|
pub fn Reader::position(self : Reader) -> Int32 {
  self.pos
}

///|
pub fn Reader::remaining(self : Reader) -> Int32 {
  self.limit - self.pos
}

///|
pub fn Reader::sub(
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
pub fn Reader::skip(self : Reader, n : Int32) -> Unit raise Hdf5Error {
  self.require(n)
  self.pos = self.pos + n
}

///|
pub fn Reader::read_u8(self : Reader) -> Byte raise Hdf5Error {
  self.require(1)
  let value = self.bytes.unsafe_get(self.pos)
  self.pos = self.pos + 1
  value
}

///|
pub fn Reader::read_u16_le(self : Reader) -> UInt16 raise Hdf5Error {
  self.require(2)
  let value = self.bytes.unsafe_read_uint16_le(self.pos)
  self.pos = self.pos + 2
  value
}

///|
pub fn Reader::read_u32_le(self : Reader) -> UInt32 raise Hdf5Error {
  self.require(4)
  let value = self.bytes.unsafe_read_uint32_le(self.pos)
  self.pos = self.pos + 4
  value
}

///|
pub fn Reader::read_u64_le(self : Reader) -> UInt64 raise Hdf5Error {
  self.require(8)
  let value = self.bytes.unsafe_read_uint64_le(self.pos)
  self.pos = self.pos + 8
  value
}

///|
/// Reads an unsigned little-endian integer of 1, 2, 4 or 8 bytes.
pub fn Reader::read_uint_le(
  self : Reader,
  size : Int32,
) -> UInt64 raise Hdf5Error {
  match size {
    1 => self.read_u8().to_uint64()
    2 => self.read_u16_le().to_uint64()
    4 => self.read_u32_le().to_uint64()
    8 => self.read_u64_le()
    _ => raise UnsupportedAddressSize(size)
  }
}

///|
pub fn Reader::read_bytes(self : Reader, n : Int32) -> Bytes raise Hdf5Error {
  self.require(n)
  let view = self.bytes.get_view(start=self.pos, end=self.pos + n)
  let result = match view {
    Some(view) => view.to_owned()
    None => raise UnexpectedEof(self.pos.to_int64())
  }
  self.pos = self.pos + n
  result
}

///|
/// Reads `n` bytes as a UTF-8 string without validating the encoding.
pub fn Reader::read_fixed_string(
  self : Reader,
  n : Int32,
) -> String raise Hdf5Error {
  self.require(n)
  let view = self.bytes.get_view(start=self.pos, end=self.pos + n).unwrap()
  let result = @utf8.decode(view) catch {
    _ => raise InvalidField("invalid UTF-8 string")
  }
  self.pos = self.pos + n
  result
}
