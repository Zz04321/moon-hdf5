///|
/// An attribute attached to an HDF5 object.
pub(all) struct Attribute {
  name : String
  dataspace : Dataspace
  datatype : Datatype
  values : DatasetValues?
} derive(@debug.Debug)

///|
pub extend Attribute with @debug.Debug::{to_repr}

///|
fn decode_c_string(bytes : Bytes) -> String raise Hdf5Error {
  let mut end = 0
  while end < bytes.length() && bytes.unsafe_get(end) != b'\x00' {
    end = end + 1
  }
  let view = bytes.get_view(start=0, end~).unwrap()
  @utf8.decode(view) catch {
    _ => raise InvalidField("invalid UTF-8 attribute name")
  }
}

///|
fn parse_attribute_message(
  message : HeaderMessage,
  bytes : Bytes,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Attribute raise Hdf5Error {
  guard message.msg_type == msg_attribute else {
    raise InvalidField("message is not an attribute message")
  }
  let reader = Reader::new(message.data)
  let version = reader.read_u8().to_int()
  let flags = if version == 1 {
    let _reserved = reader.read_u8()
    0
  } else {
    reader.read_u8().to_int()
  }
  let name_size = reader.read_u16_le().to_int()
  let datatype_size = reader.read_u16_le().to_int()
  let dataspace_size = reader.read_u16_le().to_int()
  if version >= 3 {
    let _encoding = reader.read_u8()
  }
  guard name_size >= 0 && datatype_size >= 0 && dataspace_size >= 0 else {
    raise InvalidField("negative attribute field size")
  }
  let name_bytes = reader.read_bytes(name_size)
  if version == 1 {
    reader.skip(round_up_to_eight(name_size) - name_size)
  }
  let datatype_data = reader.read_bytes(datatype_size)
  if version == 1 {
    reader.skip(round_up_to_eight(datatype_size) - datatype_size)
  }
  let dataspace_data = reader.read_bytes(dataspace_size)
  let data = reader.read_bytes(reader.remaining())
  guard (flags & 0x03) == 0 else {
    raise UnsupportedFeature("shared attribute datatype or dataspace")
  }
  let datatype = parse_datatype_data(datatype_data)
  let dataspace = parse_dataspace_data(dataspace_data, length_size)
  let count = dataspace.element_count()
  let values = Some(
    decode_values(
      bytes, data, count, datatype, offset_size, length_size, base_address,
    ),
  ) catch {
    UnsupportedFeature(_) => None
    error => raise error
  }
  { name: decode_c_string(name_bytes), dataspace, datatype, values, }
}

///|
/// Parses all attributes attached to an object header.
pub fn read_attributes(
  bytes : Bytes,
  header : ObjectHeader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Array[Attribute] raise Hdf5Error {
  if header.first(msg_attribute_info) is Some(message) &&
    message.data.length() >= 2 &&
    (message.data.unsafe_get(1).to_int() & 0x01) != 0 {
    raise UnsupportedFeature("dense attribute storage is not supported yet")
  }
  let attributes : Array[Attribute] = []
  for message in header.messages {
    if message.msg_type == msg_attribute {
      attributes.push(
        parse_attribute_message(
          message, bytes, offset_size, length_size, base_address,
        ),
      )
    }
  }
  attributes
}
