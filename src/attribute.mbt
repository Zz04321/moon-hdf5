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
fn parse_attribute_data(
  data : Bytes,
  message_flags : Byte,
  bytes : Bytes,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Attribute raise Hdf5Error {
  guard (message_flags.to_int() & 0x02) == 0 else {
    raise UnsupportedFeature("shared attribute messages are not supported yet")
  }
  let reader = Reader::new(data)
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
  let value_data = reader.read_bytes(reader.remaining())
  guard (flags & 0x03) == 0 else {
    raise UnsupportedFeature("shared attribute datatype or dataspace")
  }
  let datatype = parse_datatype_data(datatype_data)
  let dataspace = parse_dataspace_data(dataspace_data, length_size)
  let count = dataspace.element_count()
  let values = Some(
    decode_values(
      bytes, value_data, count, datatype, offset_size, length_size, base_address,
    ),
  ) catch {
    UnsupportedFeature(_) => None
    error => raise error
  }
  { name: decode_c_string(name_bytes), dataspace, datatype, values, }
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
  parse_attribute_data(
    message.data,
    message.flags,
    bytes,
    offset_size,
    length_size,
    base_address,
  )
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
  let attributes : Array[Attribute] = []
  for message in header.find(msg_attribute_info) {
    let reader = Reader::new(message.data)
    guard reader.read_u8() == b'\x00' else {
      raise UnsupportedFeature("unsupported attribute info message version")
    }
    let flags = reader.read_u8().to_int()
    guard (flags & 0xFC) == 0 && (flags & 3) != 2 else {
      raise InvalidField("invalid attribute info flags")
    }
    if (flags & 1) != 0 {
      reader.skip(2)
    }
    let heap_address = reader.read_uint_le(offset_size)
    let name_index_address = reader.read_uint_le(offset_size)
    let undefined = if offset_size == 8 {
      0xFFFFFFFFFFFFFFFFUL
    } else {
      (1UL << (offset_size * 8)) - 1UL
    }
    let order_index_address = if (flags & 2) != 0 {
      reader.read_uint_le(offset_size)
    } else {
      undefined
    }
    if heap_address != undefined {
      let ordered = (flags & 2) != 0 && order_index_address != undefined
      let index_address = if ordered {
        order_index_address
      } else if name_index_address != undefined {
        name_index_address
      } else {
        raise InvalidField("dense attributes have no name or order index")
      }
      let heap = parse_fractal_heap(
        bytes, heap_address, offset_size, length_size, base_address,
      )
      let records = read_metadata_btree_records(
        bytes,
        index_address,
        offset_size,
        length_size,
        base_address,
        [8, 9],
      )
      let dense : Array[Attribute] = []
      for record in records {
        dense.push(
          parse_attribute_data(
            heap.get_data(record.heap_id),
            record.flags,
            bytes,
            offset_size,
            length_size,
            base_address,
          ),
        )
      }
      if !ordered {
        dense.sort_by((left, right) => {
          if left.name < right.name {
            -1
          } else if left.name > right.name {
            1
          } else {
            0
          }
        })
      }
      attributes.append(dense)
    } else {
      guard name_index_address == undefined && order_index_address == undefined else {
        raise InvalidField("attribute index is present without a fractal heap")
      }
    }
  }
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
