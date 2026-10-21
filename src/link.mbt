///|
fn read_link_name(reader : Reader, length : Int32) -> String raise Hdf5Error {
  let bytes = reader.read_bytes(length)
  let view = bytes.get_view().unwrap()
  @utf8.decode(view) catch {
    _ => raise InvalidField("invalid UTF-8 link name")
  }
}

///|
/// Parses a version 1 link message.
fn parse_link_message(
  message : HeaderMessage,
  offset_size : Int32,
) -> GroupLink raise Hdf5Error {
  guard message.msg_type == msg_link else {
    raise InvalidField("message is not a link message")
  }
  let reader = Reader::new(message.data)
  let version = reader.read_u8().to_int()
  guard version == 1 else {
    raise InvalidField("unsupported link message version")
  }
  let flags = reader.read_u8().to_int()
  let link_type = if (flags & 0x08) != 0 {
    reader.read_u8().to_int()
  } else {
    0
  }
  if (flags & 0x04) != 0 {
    reader.skip(8)
  }
  if (flags & 0x10) != 0 {
    let _character_set = reader.read_u8()
  }
  let length_size = 1 << (flags & 0x03)
  let name_length = reader.read_uint_le(length_size).to_int()
  guard name_length >= 0 else {
    raise InvalidField("negative link name length")
  }
  let name = read_link_name(reader, name_length)
  match link_type {
    0 => {
      let address = reader.read_uint_le(offset_size)
      {
        name,
        object_header_address: address,
        cache_type: 0U,
        link_target: None,
      }
    }
    1 => {
      let target_length = reader.read_u16_le().to_int()
      let target = read_link_name(reader, target_length)
      {
        name,
        object_header_address: 0UL,
        cache_type: 2U,
        link_target: Some(target),
      }
    }
    64 => {
      let target_length = reader.read_u16_le().to_int()
      let target = read_link_name(reader, target_length)
      {
        name,
        object_header_address: 0UL,
        cache_type: 64U,
        link_target: Some(target),
      }
    }
    _ => raise InvalidField("unsupported link type")
  }
}

///|
/// Reads a link message stored in a dense group's fractal heap.
fn parse_dense_link(
  data : Bytes,
  flags : Byte,
  offset_size : Int32,
) -> GroupLink raise Hdf5Error {
  parse_link_message({ msg_type: msg_link, flags, data, }, offset_size)
}

///|
/// Reads all hard, soft and external links stored in a v2 group, including
/// compact links and links indexed by a fractal heap and v2 B-tree.
fn read_group_links_v2(
  bytes : Bytes,
  header : ObjectHeader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Array[GroupLink] raise Hdf5Error {
  let sources : Array[&Hdf5Source] = [bytes]
  read_source_group_links_v2(
    sources[0],
    header,
    offset_size,
    length_size,
    base_address,
  )
}

///|
fn read_source_group_links_v2(
  source : &Hdf5Source,
  header : ObjectHeader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Array[GroupLink] raise Hdf5Error {
  let links : Array[GroupLink] = []
  for message in header.messages {
    if message.msg_type == msg_link_info {
      let reader = Reader::new(message.data)
      guard reader.read_u8() == b'\x00' else {
        raise UnsupportedFeature("unsupported link info message version")
      }
      let flags = reader.read_u8().to_int()
      guard (flags & 0xFC) == 0 && (flags & 0x03) != 2 else {
        raise InvalidField("invalid link info flags")
      }
      if (flags & 0x01) != 0 {
        reader.skip(8)
      }
      let heap_address = reader.read_uint_le(offset_size)
      let name_index_address = reader.read_uint_le(offset_size)
      let undefined = if offset_size == 8 {
        0xFFFFFFFFFFFFFFFFUL
      } else {
        (1UL << (offset_size * 8)) - 1UL
      }
      let order_index_address = if (flags & 0x02) != 0 {
        reader.read_uint_le(offset_size)
      } else {
        undefined
      }
      if heap_address != undefined {
        let ordered = (flags & 0x02) != 0 && order_index_address != undefined
        let index_address = if ordered {
          order_index_address
        } else if name_index_address != undefined {
          name_index_address
        } else {
          raise InvalidField("dense group has no name or order index")
        }
        let heap = parse_source_fractal_heap(
          source, heap_address, offset_size, length_size, base_address,
        )
        let records = read_source_metadata_btree_records(
          source,
          index_address,
          offset_size,
          length_size,
          base_address,
          [5, 6],
        )
        let dense : Array[GroupLink] = []
        for record in records {
          dense.push(
            parse_dense_link(
              heap.get_data(record.heap_id),
              record.flags,
              offset_size,
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
        links.append(dense)
      } else {
        guard name_index_address == undefined &&
          order_index_address == undefined else {
          raise InvalidField("group index is present without a fractal heap")
        }
      }
    }
  }
  for message in header.messages {
    if message.msg_type == msg_link {
      links.push(parse_link_message(message, offset_size))
    }
  }
  links
}
