///|
/// Size of a metadata checksum stored on disk.
let metadata_checksum_size : Int32 = 4

///|
fn checksum_rot(value : UInt32, bits : Int32) -> UInt32 {
  (value << bits) | (value >> (32 - bits))
}

///|
fn checksum_mix(
  a_value : UInt32,
  b_value : UInt32,
  c_value : UInt32,
) -> (UInt32, UInt32, UInt32) {
  let mut a = a_value
  let mut b = b_value
  let mut c = c_value
  a = a - c
  a = a ^ checksum_rot(c, 4)
  c = c + b
  b = b - a
  b = b ^ checksum_rot(a, 6)
  a = a + c
  c = c - b
  c = c ^ checksum_rot(b, 8)
  b = b + a
  a = a - c
  a = a ^ checksum_rot(c, 16)
  c = c + b
  b = b - a
  b = b ^ checksum_rot(a, 19)
  a = a + c
  c = c - b
  c = c ^ checksum_rot(b, 4)
  b = b + a
  (a, b, c)
}

///|
fn checksum_final(
  a_value : UInt32,
  b_value : UInt32,
  c_value : UInt32,
) -> UInt32 {
  let mut a = a_value
  let mut b = b_value
  let mut c = c_value
  c = c ^ b
  c = c - checksum_rot(b, 14)
  a = a ^ c
  a = a - checksum_rot(c, 11)
  b = b ^ a
  b = b - checksum_rot(a, 25)
  c = c ^ b
  c = c - checksum_rot(b, 16)
  a = a ^ c
  a = a - checksum_rot(c, 4)
  b = b ^ a
  b = b - checksum_rot(a, 14)
  c = c ^ b
  c - checksum_rot(b, 24)
}

///|
fn checksum_byte(
  bytes : Bytes,
  start : Int32,
  zero_start : Int32,
  zero_length : Int32,
  index : Int32,
) -> UInt32 {
  if index >= zero_start && index < zero_start + zero_length {
    0U
  } else {
    bytes.unsafe_get(start + index).to_int().reinterpret_as_uint()
  }
}

///|
/// Bob Jenkins lookup3, matching H5_checksum_lookup3.
///
/// `zero_start` and `zero_length` describe a range inside the input that is
/// treated as zero. Fractal heap direct blocks need this because their
/// checksum covers the whole block while the stored checksum field itself is
/// zeroed during computation.
fn checksum_lookup3(
  bytes : Bytes,
  start : Int32,
  length : Int32,
  zero_start : Int32,
  zero_length : Int32,
  initval : UInt32,
) -> UInt32 {
  let mut a = 0xDEADBEEFU + length.reinterpret_as_uint() + initval
  let mut b = a
  let mut c = a
  let mut remaining = length
  let mut offset = 0
  while remaining > 12 {
    a = a +
      checksum_byte(bytes, start, zero_start, zero_length, offset) +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 1) << 8) +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 2) << 16) +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 3) << 24)
    b = b +
      checksum_byte(bytes, start, zero_start, zero_length, offset + 4) +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 5) << 8) +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 6) << 16) +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 7) << 24)
    c = c +
      checksum_byte(bytes, start, zero_start, zero_length, offset + 8) +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 9) << 8) +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 10) << 16) +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 11) << 24)
    let (next_a, next_b, next_c) = checksum_mix(a, b, c)
    a = next_a
    b = next_b
    c = next_c
    remaining = remaining - 12
    offset = offset + 12
  }
  if remaining >= 1 {
    a = a + checksum_byte(bytes, start, zero_start, zero_length, offset)
  }
  if remaining >= 2 {
    a = a +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 1) << 8)
  }
  if remaining >= 3 {
    a = a +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 2) << 16)
  }
  if remaining >= 4 {
    a = a +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 3) << 24)
  }
  if remaining >= 5 {
    b = b + checksum_byte(bytes, start, zero_start, zero_length, offset + 4)
  }
  if remaining >= 6 {
    b = b +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 5) << 8)
  }
  if remaining >= 7 {
    b = b +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 6) << 16)
  }
  if remaining >= 8 {
    b = b +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 7) << 24)
  }
  if remaining >= 9 {
    c = c + checksum_byte(bytes, start, zero_start, zero_length, offset + 8)
  }
  if remaining >= 10 {
    c = c +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 9) << 8)
  }
  if remaining >= 11 {
    c = c +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 10) << 16)
  }
  if remaining >= 12 {
    c = c +
      (checksum_byte(bytes, start, zero_start, zero_length, offset + 11) << 24)
  }
  if remaining == 0 {
    return c
  }
  checksum_final(a, b, c)
}

///|
fn metadata_checksum(bytes : Bytes, start : Int32, length : Int32) -> UInt32 {
  checksum_lookup3(bytes, start, length, 0, 0, 0U)
}

///|
fn stored_metadata_checksum(
  bytes : Bytes,
  offset : Int32,
) -> UInt32 raise Hdf5Error {
  guard offset >= 0 && offset.to_int64() + 4L <= bytes.length().to_int64() else {
    raise UnexpectedEof(offset.to_int64())
  }
  bytes.unsafe_read_uint32_le(offset)
}

///|
/// Verifies a checksum stored directly after the covered bytes.
fn verify_metadata_checksum(
  bytes : Bytes,
  start : Int32,
  end : Int32,
  label : String,
) -> Unit raise Hdf5Error {
  guard start >= 0 &&
    end >= start &&
    end.to_int64() + 4L <= bytes.length().to_int64() else {
    raise UnexpectedEof(end.to_int64())
  }
  let stored = stored_metadata_checksum(bytes, end)
  let computed = metadata_checksum(bytes, start, end - start)
  guard stored == computed else {
    raise InvalidField(label + " checksum mismatch")
  }
}

///|
/// Verifies a checksum inside the covered range whose bytes are zero during
/// computation. Fractal heap direct blocks use this layout.
fn verify_metadata_checksum_zeroed(
  bytes : Bytes,
  start : Int32,
  length : Int32,
  checksum_offset : Int32,
  label : String,
) -> Unit raise Hdf5Error {
  guard start >= 0 &&
    length >= 4 &&
    checksum_offset >= start &&
    checksum_offset.to_int64() + 4L <= start.to_int64() + length.to_int64() &&
    start.to_int64() + length.to_int64() <= bytes.length().to_int64() else {
    raise UnexpectedEof(start.to_int64())
  }
  let stored = stored_metadata_checksum(bytes, checksum_offset)
  let computed = checksum_lookup3(
    bytes,
    start,
    length,
    checksum_offset - start,
    metadata_checksum_size,
    0U,
  )
  guard stored == computed else {
    raise InvalidField(label + " checksum mismatch")
  }
}
