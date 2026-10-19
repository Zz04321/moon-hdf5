///|
/// The HDF5 on-disk signature.
let hdf5_signature : Bytes = b"\x89HDF\r\n\x1a\n"

///|
fn signature_at(bytes : Bytes, offset : Int32) -> Bool {
  guard offset >= 0 && offset + 8 <= bytes.length() else { false }
  match bytes.get_view(start=offset, end=offset + 8) {
    Some(view) => view.equal_to_bytes(hdf5_signature)
    None => false
  }
}

///|
/// Locates the HDF5 superblock, allowing for a user block before it.
///
/// HDF5 places the superblock at byte 0 or at a power-of-two offset of at
/// least 512 bytes. The scan stops after 1 MiB, matching libhdf5's practical
/// user-block limit.
pub fn find_superblock_offset(bytes : Bytes) -> Int32 raise Hdf5Error {
  if signature_at(bytes, 0) {
    return 0
  }
  let mut offset = 512
  while offset <= 1048576 && offset + 8 <= bytes.length() {
    if signature_at(bytes, offset) {
      return offset
    }
    offset = offset * 2
  }
  raise InvalidSignature
}

///|
/// Parsed HDF5 superblock.
pub(all) struct Superblock {
  superblock_offset : Int32
  superblock_end : Int32
  version : Int32
  offset_size : Int32
  length_size : Int32
  file_consistency_flags : UInt32
  base_address : UInt64
  free_space_address : UInt64
  end_of_file_address : UInt64
  driver_info_address : UInt64
  superblock_extension_address : UInt64
  root_group_address : UInt64
  root_group_object_header : UInt64
  group_leaf_node_k : UInt16
  group_internal_node_k : UInt16
  superblock_checksum : UInt32
} derive(Eq, @debug.Debug)

///|
pub extend Superblock with Eq::{not_equal, equal}

///|
pub extend Superblock with @debug.Debug::{to_repr}

///|
fn validate_address_sizes(
  offset_size : Int32,
  length_size : Int32,
) -> Unit raise Hdf5Error {
  guard offset_size == 2 || offset_size == 4 || offset_size == 8 else {
    raise UnsupportedAddressSize(offset_size)
  }
  guard length_size == 2 || length_size == 4 || length_size == 8 else {
    raise UnsupportedAddressSize(length_size)
  }
}

///|
fn parse_superblock_v0(
  reader : Reader,
  offset : Int32,
  version : Int32,
) -> Superblock raise Hdf5Error {
  let _free_storage_version = reader.read_u8()
  let _root_group_version = reader.read_u8()
  let _reserved_0 = reader.read_u8()
  let _shared_header_version = reader.read_u8()
  let offset_size = reader.read_u8().to_int()
  let length_size = reader.read_u8().to_int()
  let _reserved_1 = reader.read_u8()
  validate_address_sizes(offset_size, length_size)
  let group_leaf_node_k = reader.read_u16_le()
  let group_internal_node_k = reader.read_u16_le()
  let file_consistency_flags = reader.read_u32_le()
  let base_address = reader.read_uint_le(offset_size)
  let free_space_address = reader.read_uint_le(offset_size)
  let end_of_file_address = reader.read_uint_le(offset_size)
  let driver_info_address = reader.read_uint_le(offset_size)
  let _root_link_name_offset = reader.read_uint_le(offset_size)
  let root_group_object_header = reader.read_uint_le(offset_size)
  let _root_cache_type = reader.read_u32_le()
  let _root_reserved = reader.read_u32_le()
  let _root_scratch = reader.read_bytes(16)
  {
    superblock_offset: offset,
    superblock_end: reader.position(),
    version,
    offset_size,
    length_size,
    file_consistency_flags,
    base_address,
    free_space_address,
    end_of_file_address,
    driver_info_address,
    superblock_extension_address: 0UL,
    root_group_address: root_group_object_header,
    root_group_object_header,
    group_leaf_node_k,
    group_internal_node_k,
    superblock_checksum: 0U,
  }
}

///|
fn parse_superblock_v2(
  reader : Reader,
  offset : Int32,
  version : Int32,
) -> Superblock raise Hdf5Error {
  let offset_size = reader.read_u8().to_int()
  let length_size = reader.read_u8().to_int()
  let file_consistency_flags = reader.read_u8().to_int().reinterpret_as_uint()
  validate_address_sizes(offset_size, length_size)
  let base_address = reader.read_uint_le(offset_size)
  let superblock_extension_address = reader.read_uint_le(offset_size)
  let end_of_file_address = reader.read_uint_le(offset_size)
  let root_group_address = reader.read_uint_le(offset_size)
  let superblock_checksum = reader.read_u32_le()
  {
    superblock_offset: offset,
    superblock_end: reader.position(),
    version,
    offset_size,
    length_size,
    file_consistency_flags,
    base_address,
    free_space_address: 0UL,
    end_of_file_address,
    driver_info_address: 0UL,
    superblock_extension_address,
    root_group_address,
    root_group_object_header: root_group_address,
    group_leaf_node_k: 0,
    group_internal_node_k: 0,
    superblock_checksum,
  }
}

///|
/// Parses the HDF5 superblock, including a leading user block if present.
pub fn parse_superblock(bytes : Bytes) -> Superblock raise Hdf5Error {
  let offset = find_superblock_offset(bytes)
  let reader = Reader::at(bytes, offset)
  let _signature = reader.read_bytes(8)
  let version = reader.read_u8().to_int()
  match version {
    0 | 1 => parse_superblock_v0(reader, offset, version)
    2 | 3 => parse_superblock_v2(reader, offset, version)
    _ => raise UnsupportedSuperblockVersion(version)
  }
}
