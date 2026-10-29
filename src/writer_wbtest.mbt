///|
test "writer round-trips groups datasets and attributes" {
  let writer = Hdf5Writer::new()
  writer.add_group("/science/results")
  writer.add_dataset(
    "/science/results/counts",
    [2UL, 2UL],
    Int64Values([1L, -2L, 3L, 4L]),
  )
  writer.add_dataset("/weight", [], Float64Values([1.5]))
  writer.add_dataset("/labels", [2UL], FixedStringValues(["moon", "数据"], 8))
  writer.add_dataset("/empty", [0UL, 3UL], UInt64Values([]))
  writer.add_attribute("/", "title", [], FixedStringValues(["writer demo"], 12))
  writer.add_attribute("/science", "version", [], UInt64Values([1UL]))
  writer.add_attribute(
    "/science/results/counts",
    "unit",
    [],
    FixedStringValues(["items"], 8),
  )

  let file = Hdf5File::parse(writer.encode())
  assert_eq(file.superblock.version, 3)
  assert_eq(file.walk().map(node => node.path), [
    "/science", "/science/results", "/science/results/counts", "/weight", "/labels",
    "/empty",
  ])
  assert_eq(integer_values(file.read_dataset("/science/results/counts")), [
    1L, -2L, 3L, 4L,
  ])
  assert_eq(float_values(file.read_dataset("/weight")), [1.5])
  assert_eq(string_values(file.read_dataset("/labels")), ["moon", "数据"])
  assert_eq(unsigned_values(file.read_dataset("/empty")), [])
  match file.attributes("/")[0].values {
    Some(Strings(values)) => assert_eq(values, ["writer demo"])
    _ => abort("expected root string attribute")
  }
  match file.attributes("/science")[0].values {
    Some(UnsignedIntegers(values)) => assert_eq(values, [1UL])
    _ => abort("expected group integer attribute")
  }
  match file.attributes("/science/results/counts")[0].values {
    Some(Strings(values)) => assert_eq(values, ["items"])
    _ => abort("expected dataset string attribute")
  }
}

///|
test "writer uses dense links and attributes above the compact threshold" {
  let writer = Hdf5Writer::new()
  writer.add_group("/dense")
  for index in 0..<9 {
    writer.add_dataset(
      "/dense/value_" + index.to_string(),
      [],
      UInt64Values([index.to_uint64()]),
    )
    writer.add_attribute(
      "/dense",
      "dense_attr_" + index.to_string(),
      [],
      UInt64Values([(index + 10).to_uint64()]),
    )
  }
  writer.add_external_link("/dense", "external", "target.h5", "/value_0")
  writer.add_attribute(
    "/dense",
    "dense_text",
    [],
    VariableStringValues(["密集属性"]),
  )
  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  let header = parse_object_header(
    bytes,
    file.resolve("/dense").address,
    8,
    8,
    0UL,
  )
  assert_eq(file.links(header).length(), 10)
  let attributes = file.attributes("/dense")
  assert_eq(attributes.length(), 10)
  match attributes.search_by(attribute => attribute.name == "dense_text") {
    Some(index) =>
      match attributes[index].values {
        Some(Strings(values)) => assert_eq(values, ["密集属性"])
        _ => abort("expected dense variable-length string attribute")
      }
    None => abort("expected dense_text attribute")
  }
  assert_eq(unsigned_values(file.read_dataset("/dense/value_8")), [8UL])
  for message_type in [msg_link_info, msg_attribute_info] {
    let info = header.find(message_type)[0]
    let reader = Reader::new(info.data)
    reader.skip(2)
    assert_true(reader.read_uint_le(8) != 0xFFFFFFFFFFFFFFFFUL)
    assert_true(reader.read_uint_le(8) != 0xFFFFFFFFFFFFFFFFUL)
  }
  assert_true(!header.messages.any(message => message.msg_type == msg_link))
  assert_true(
    !header.messages.any(message => message.msg_type == msg_attribute),
  )

  let compact = Hdf5Writer::new()
  compact.add_group("/compact")
  for index in 0..<8 {
    compact.add_dataset(
      "/compact/value_" + index.to_string(),
      [],
      UInt64Values([index.to_uint64()]),
    )
  }
  let compact_bytes = compact.encode()
  let compact_file = Hdf5File::parse(compact_bytes)
  let compact_header = parse_object_header(
    compact_bytes,
    compact_file.resolve("/compact").address,
    8,
    8,
    0UL,
  )
  let info = compact_header.find(msg_link_info)[0]
  let info_reader = Reader::new(info.data)
  info_reader.skip(2)
  assert_eq(info_reader.read_uint_le(8), 0xFFFFFFFFFFFFFFFFUL)
  assert_eq(info_reader.read_uint_le(8), 0xFFFFFFFFFFFFFFFFUL)
}

///|
test "writer round-trips object reference datasets and attributes" {
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/refs",
    [4UL],
    ObjectReferenceValues([
      Some("/target"),
      Some("/group"),
      None,
      Some("/group/later"),
    ]),
  )
  writer.add_chunked_dataset(
    "/chunked_refs",
    [2UL],
    [1U],
    ObjectReferenceValues([Some("/target"), None]),
  )
  writer.add_dataset("/target", [2UL], Int64Values([1L, 2L]))
  writer.add_attribute(
    "/target",
    "self_ref",
    [],
    ObjectReferenceValues([Some("/target")]),
  )
  writer.add_group("/group")
  writer.add_dataset("/group/later", [], UInt64Values([7UL]))

  let file = Hdf5File::parse(writer.encode())
  let dataset = file.read_dataset("/refs")
  assert_eq(dataset.datatype.class, 7)
  assert_eq(dataset.datatype.reference_kind, 0)
  assert_eq(dataset.datatype.size, 8)
  let references = match dataset.values {
    Structured(values) => values
    _ => abort("expected structured object references")
  }
  assert_eq(references.length(), 4)
  let target = file.resolve("/target")
  let group = file.resolve("/group")
  let later = file.resolve("/group/later")
  match references[0] {
    Reference(bytes) =>
      assert_eq(
        file.resolve_object_reference(bytes),
        Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
      )
    _ => abort("expected target object reference")
  }
  match references[1] {
    Reference(bytes) =>
      assert_eq(
        file.resolve_object_reference(bytes),
        Some({ address: group.address, kind: Group, path: Some("/group"), }),
      )
    _ => abort("expected group object reference")
  }
  match references[2] {
    Reference(bytes) => assert_eq(file.resolve_object_reference(bytes), None)
    _ => abort("expected null object reference")
  }
  match references[3] {
    Reference(bytes) =>
      assert_eq(
        file.resolve_object_reference(bytes),
        Some({
          address: later.address,
          kind: Dataset,
          path: Some("/group/later"),
        }),
      )
    _ => abort("expected forward object reference")
  }
  let chunked = match file.read_dataset("/chunked_refs").values {
    Structured([Reference(first), Reference(second)]) => (first, second)
    _ => abort("expected chunked object references")
  }
  assert_eq(
    file.resolve_object_reference(chunked.0),
    Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
  )
  assert_eq(file.resolve_object_reference(chunked.1), None)
  let attributes = file.attributes("/target")
  let self_ref = attributes.search_by(attribute => attribute.name == "self_ref")
  match self_ref {
    Some(index) =>
      match attributes[index].values {
        Some(Structured([Reference(bytes)])) =>
          assert_eq(
            file.resolve_object_reference(bytes),
            Some({
              address: target.address,
              kind: Dataset,
              path: Some("/target"),
            }),
          )
        _ => abort("expected self reference attribute")
      }
    None => abort("expected self_ref attribute")
  }
}

///|
test "writer writes object references nested in structured values" {
  let writer = Hdf5Writer::new()
  let object_ref = Datatype::write_object_reference()
  let pair_type = Datatype::write_array(object_ref, [2UL])
  let history_type = Datatype::write_vlen_sequence(object_ref)
  let record_type = Datatype::write_compound([
    ("label", Datatype::write_fixed_string(6)),
    ("target", object_ref),
    ("pair", pair_type),
    ("history", history_type),
  ])
  writer.add_dataset("/target", [1UL], Int64Values([5L]))
  writer.add_group("/group")
  writer.add_dataset("/group/later", [], UInt64Values([9UL]))
  let first = Compound([
    { name: "label", value: Text("first"), },
    { name: "target", value: ObjectReference(Some("/target")), },
    {
      name: "pair",
      value: Elements([ObjectReference(Some("/group")), ObjectReference(None)]),
    },
    {
      name: "history",
      value: Sequence([
        ObjectReference(Some("/target")),
        ObjectReference(Some("/group/later")),
      ]),
    },
  ])
  let second = Compound([
    { name: "label", value: Text("second"), },
    { name: "target", value: ObjectReference(Some("/group/later")), },
    {
      name: "pair",
      value: Elements([ObjectReference(None), ObjectReference(Some("/group"))]),
    },
    { name: "history", value: Sequence([]), },
  ])
  writer.add_dataset(
    "/records",
    [2UL],
    StructuredValues(record_type, [first, second]),
  )
  writer.add_attribute(
    "/records",
    "prototype",
    [],
    StructuredValues(record_type, [first]),
  )

  let file = Hdf5File::parse(writer.encode())
  let target = file.resolve("/target")
  let group = file.resolve("/group")
  let later = file.resolve("/group/later")
  let records = match file.read_dataset("/records").values {
    Structured(values) => values
    _ => abort("expected structured reference records")
  }
  assert_eq(records.length(), 2)
  let first_fields = match records[0] {
    Compound(fields) => fields
    _ => abort("expected first reference compound")
  }
  assert_eq(first_fields[0].value, Text("first"))
  match first_fields[1].value {
    Reference(bytes) =>
      assert_eq(
        file.resolve_object_reference(bytes),
        Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
      )
    _ => abort("expected compound object reference")
  }
  match first_fields[2].value {
    Elements([Reference(first_ref), Reference(null_ref)]) => {
      assert_eq(
        file.resolve_object_reference(first_ref),
        Some({ address: group.address, kind: Group, path: Some("/group"), }),
      )
      assert_eq(file.resolve_object_reference(null_ref), None)
    }
    _ => abort("expected reference array")
  }
  match first_fields[3].value {
    Sequence([Reference(first_ref), Reference(forward_ref)]) => {
      assert_eq(
        file.resolve_object_reference(first_ref),
        Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
      )
      assert_eq(
        file.resolve_object_reference(forward_ref),
        Some({
          address: later.address,
          kind: Dataset,
          path: Some("/group/later"),
        }),
      )
    }
    _ => abort("expected reference sequence")
  }
  let second_fields = match records[1] {
    Compound(fields) => fields
    _ => abort("expected second reference compound")
  }
  match second_fields[1].value {
    Reference(bytes) =>
      assert_eq(
        file.resolve_object_reference(bytes),
        Some({
          address: later.address,
          kind: Dataset,
          path: Some("/group/later"),
        }),
      )
    _ => abort("expected forward compound object reference")
  }
  match second_fields[3].value {
    Sequence([]) => ()
    _ => abort("expected empty reference sequence")
  }
  let attributes = file.attributes("/records")
  match attributes.search_by(attribute => attribute.name == "prototype") {
    Some(index) =>
      match attributes[index].values {
        Some(Structured([Compound(fields)])) =>
          match fields[1].value {
            Reference(bytes) =>
              assert_eq(
                file.resolve_object_reference(bytes),
                Some({
                  address: target.address,
                  kind: Dataset,
                  path: Some("/target"),
                }),
              )
            _ => abort("expected prototype object reference")
          }
        _ => abort("expected structured prototype attribute")
      }
    None => abort("expected prototype attribute")
  }
}

///|
fn writer_region_integer_values(
  file : Hdf5File,
  reference : Bytes,
) -> Array[Int64] raise Hdf5Error {
  match file.read_region_reference(reference) {
    Some(result) =>
      match result.values {
        Integers(values) => values
        _ => abort("expected region reference integers")
      }
    None => abort("expected a non-null region reference")
  }
}

///|
test "writer round-trips region reference datasets" {
  let writer = Hdf5Writer::new()
  let region_ref = Datatype::write_region_reference()
  writer.add_dataset(
    "/region_refs",
    [7UL],
    StructuredValues(region_ref, [
      RegionReference(None, RegionEmpty),
      RegionReference(Some("/source"), RegionPoints([[4UL], [0UL], [3UL]])),
      RegionReference(
        Some("/source"),
        RegionHyperslab([1UL], [1UL], [1UL], [2UL]),
      ),
      RegionReference(
        Some("/source"),
        RegionHyperslab([0UL], [3UL], [2UL], [2UL]),
      ),
      RegionReference(
        Some("/source"),
        RegionBlocks([
          { start: [0UL], end: [1UL], },
          { start: [4UL], end: [4UL], },
        ]),
      ),
      RegionReference(Some("/source"), RegionAll),
      RegionReference(
        Some("/matrix"),
        RegionHyperslab([0UL, 0UL], [2UL, 3UL], [2UL, 2UL], [2UL, 1UL]),
      ),
    ]),
  )
  writer.add_dataset("/source", [6UL], Int64Values([0L, 1L, 2L, 3L, 4L, 5L]))
  writer.add_dataset(
    "/matrix",
    [6UL, 5UL],
    Int64Values([
      0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L, 16L,
      17L, 18L, 19L, 20L, 21L, 22L, 23L, 24L, 25L, 26L, 27L, 28L, 29L,
    ]),
  )

  let file = Hdf5File::parse(writer.encode())
  let dataset = file.read_dataset("/region_refs")
  assert_eq(dataset.datatype.class, 7)
  assert_eq(dataset.datatype.reference_kind, 1)
  assert_eq(dataset.datatype.size, 12)
  let references = match dataset.values {
    Structured(values) => values
    _ => abort("expected structured region references")
  }
  assert_eq(references.length(), 7)
  match references[0] {
    Reference(bytes) =>
      assert_true(file.resolve_region_reference(bytes) is None)
    _ => abort("expected null region reference")
  }
  match references[1] {
    Reference(bytes) => {
      let target = match file.resolve_region_reference(bytes) {
        Some(target) => target
        None => abort("expected points region reference")
      }
      assert_eq(target.path, Some("/source"))
      assert_eq(target.selection, RegionPoints([[4UL], [0UL], [3UL]]))
      assert_eq(writer_region_integer_values(file, bytes), [4L, 0L, 3L])
    }
    _ => abort("expected points region reference bytes")
  }
  match references[2] {
    Reference(bytes) =>
      assert_eq(writer_region_integer_values(file, bytes), [1L, 2L])
    _ => abort("expected slab region reference")
  }
  match references[3] {
    Reference(bytes) =>
      assert_eq(writer_region_integer_values(file, bytes), [0L, 1L, 3L, 4L])
    _ => abort("expected regular region reference")
  }
  match references[4] {
    Reference(bytes) =>
      assert_eq(writer_region_integer_values(file, bytes), [0L, 1L, 4L])
    _ => abort("expected block region reference")
  }
  match references[5] {
    Reference(bytes) =>
      assert_eq(writer_region_integer_values(file, bytes), [
        0L, 1L, 2L, 3L, 4L, 5L,
      ])
    _ => abort("expected all region reference")
  }
  match references[6] {
    Reference(bytes) =>
      assert_eq(writer_region_integer_values(file, bytes), [
        0L, 3L, 5L, 8L, 10L, 13L, 15L, 18L,
      ])
    _ => abort("expected matrix region reference")
  }
}

///|
test "writer writes region references nested in structured values" {
  let writer = Hdf5Writer::new()
  let region_ref = Datatype::write_region_reference()
  let pair_type = Datatype::write_array(region_ref, [2UL])
  let record_type = Datatype::write_compound([
    ("label", Datatype::write_fixed_string(6)),
    ("target", region_ref),
    ("pair", pair_type),
  ])
  let record = Compound([
    { name: "label", value: Text("first"), },
    {
      name: "target",
      value: RegionReference(Some("/source"), RegionPoints([[2UL]])),
    },
    {
      name: "pair",
      value: Elements([
        RegionReference(
          Some("/source"),
          RegionHyperslab([1UL], [1UL], [1UL], [2UL]),
        ),
        RegionReference(None, RegionEmpty),
      ]),
    },
  ])
  writer.add_dataset("/records", [1UL], StructuredValues(record_type, [record]))
  writer.add_chunked_dataset(
    "/chunked_regions",
    [2UL],
    [1U],
    StructuredValues(region_ref, [
      RegionReference(Some("/source"), RegionAll),
      RegionReference(None, RegionEmpty),
    ]),
  )
  writer.add_attribute(
    "/records",
    "prototype",
    [],
    StructuredValues(record_type, [record]),
  )
  writer.add_dataset("/source", [4UL], Int64Values([10L, 11L, 12L, 13L]))

  let file = Hdf5File::parse(writer.encode())
  let record_values = match file.read_dataset("/records").values {
    Structured([Compound(fields)]) => fields
    _ => abort("expected one region reference record")
  }
  match record_values[1].value {
    Reference(bytes) =>
      assert_eq(writer_region_integer_values(file, bytes), [12L])
    _ => abort("expected compound region reference")
  }
  match record_values[2].value {
    Elements([Reference(slab), Reference(null_ref)]) => {
      assert_eq(writer_region_integer_values(file, slab), [11L, 12L])
      assert_true(file.resolve_region_reference(null_ref) is None)
    }
    _ => abort("expected region reference array")
  }
  match file.read_dataset("/chunked_regions").values {
    Structured([Reference(all_ref), Reference(null_ref)]) => {
      assert_eq(writer_region_integer_values(file, all_ref), [
        10L, 11L, 12L, 13L,
      ])
      assert_true(file.resolve_region_reference(null_ref) is None)
    }
    _ => abort("expected chunked region references")
  }
  let attributes = file.attributes("/records")
  match attributes.search_by(attribute => attribute.name == "prototype") {
    Some(index) =>
      match attributes[index].values {
        Some(Structured([Compound(fields)])) =>
          match fields[1].value {
            Reference(bytes) =>
              assert_eq(writer_region_integer_values(file, bytes), [12L])
            _ => abort("expected prototype region reference")
          }
        _ => abort("expected structured prototype attribute")
      }
    None => abort("expected prototype attribute")
  }
}

///|
test "writer validates region references" {
  let wrong_kind = try {
    let writer = Hdf5Writer::new()
    writer.add_dataset(
      "/refs",
      [1UL],
      StructuredValues(Datatype::write_region_reference(), [
        ObjectReference(Some("/source")),
      ]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_kind,
    Some(InvalidArgument("region reference value has the wrong kind")),
  )

  let missing = try {
    let writer = Hdf5Writer::new()
    writer.add_dataset(
      "/refs",
      [1UL],
      StructuredValues(Datatype::write_region_reference(), [
        RegionReference(Some("/missing"), RegionAll),
      ]),
    )
    writer.encode() |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    missing,
    Some(InvalidArgument("region reference target does not exist: /missing")),
  )

  let group_target = try {
    let writer = Hdf5Writer::new()
    writer.add_group("/group")
    writer.add_dataset(
      "/refs",
      [1UL],
      StructuredValues(Datatype::write_region_reference(), [
        RegionReference(Some("/group"), RegionAll),
      ]),
    )
    writer.encode() |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    group_target,
    Some(InvalidArgument("region reference target is not a dataset: /group")),
  )

  let out_of_bounds = try {
    let writer = Hdf5Writer::new()
    writer.add_dataset("/source", [2UL], Int64Values([1L, 2L]))
    writer.add_dataset(
      "/refs",
      [1UL],
      StructuredValues(Datatype::write_region_reference(), [
        RegionReference(Some("/source"), RegionPoints([[2UL]])),
      ]),
    )
    writer.encode() |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    out_of_bounds,
    Some(InvalidField("region point is outside the dataset")),
  )

  let vlen_region = try {
    Datatype::write_vlen_sequence(Datatype::write_region_reference()) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    vlen_region,
    Some(
      InvalidArgument(
        "writer variable-length sequence base cannot contain region references",
      ),
    ),
  )

  let nested_vlen_region = try {
    let nested = Datatype::write_compound([
      ("target", Datatype::write_region_reference()),
      ("count", Datatype::write_integer(true, 4)),
    ])
    Datatype::write_vlen_sequence(nested) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    nested_vlen_region,
    Some(
      InvalidArgument(
        "writer variable-length sequence base cannot contain region references",
      ),
    ),
  )
}

///|
test "writer validates object reference paths and targets" {
  let relative = try {
    Hdf5Writer::new().add_dataset(
      "/refs",
      [1UL],
      ObjectReferenceValues([Some("target")]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(relative, Some(InvalidArgument("writer paths must be absolute")))

  let missing = try {
    let writer = Hdf5Writer::new()
    writer.add_dataset(
      "/refs",
      [1UL],
      ObjectReferenceValues([Some("/missing")]),
    )
    writer.encode() |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    missing,
    Some(InvalidArgument("object reference target does not exist: /missing")),
  )

  let wrong_kind = try {
    let writer = Hdf5Writer::new()
    writer.add_dataset(
      "/refs",
      [1UL],
      StructuredValues(Datatype::write_object_reference(), [
        Reference(b"\x00\x00\x00\x00\x00\x00\x00\x00"),
      ]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_kind,
    Some(InvalidArgument("object reference value has the wrong kind")),
  )

  let nested_missing = try {
    let writer = Hdf5Writer::new()
    let record_type = Datatype::write_compound([
      ("target", Datatype::write_object_reference()),
    ])
    writer.add_dataset(
      "/refs",
      [1UL],
      StructuredValues(record_type, [
        Compound([{ name: "target", value: ObjectReference(Some("/missing")), }]),
      ]),
    )
    writer.encode() |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    nested_missing,
    Some(InvalidArgument("object reference target does not exist: /missing")),
  )
}

///|
test "writer enforces exact dense metadata limits" {
  let entry = { name: "item", hash: 0U, data: b"", }
  assert_true(!writer_dense_entries_fit(Array::make(8, entry)))
  assert_true(writer_dense_entries_fit(Array::make(9, entry)))
  assert_true(writer_dense_entries_fit(Array::make(65535, entry)))
  assert_true(!writer_dense_entries_fit(Array::make(65536, entry)))

  let at_limit = {
    name: "item",
    hash: 0U,
    data: Bytes::makei(4096, _ => b'\x00'),
  }
  let over_limit = {
    name: "item",
    hash: 0U,
    data: Bytes::makei(4097, _ => b'\x00'),
  }
  assert_true(writer_dense_entries_fit(Array::make(9, at_limit)))
  assert_true(!writer_dense_entries_fit(Array::make(9, over_limit)))

  let btree_header = encode_writer_dense_btree_header(8, 1024, 17, 0UL, 65535)
  let reader = Reader::new(btree_header)
  reader.skip(24)
  assert_eq(reader.read_u16_le(), 65535)
}

///|
test "writer continues oversized compact links and attributes" {
  let writer = Hdf5Writer::new()
  let value = "v".repeat(9000)
  writer.add_dataset("/continued", [], UInt64Values([7UL]))
  for index in 0..<8 {
    writer.add_attribute(
      "/continued",
      "large_" + index.to_string(),
      [],
      FixedStringValues([value], 9000),
    )
  }
  writer.add_group("/continued_links")
  let target = "/" + "t".repeat(9000)
  for index in 0..<8 {
    writer.add_soft_link(
      "/continued_links",
      "link_" + index.to_string(),
      target,
    )
  }

  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  let attributes = file.attributes("/continued")
  assert_eq(attributes.length(), 8)
  match attributes[0].values {
    Some(Strings(values)) => assert_eq(values, [value])
    _ => abort("expected initial fixed-string attribute")
  }
  match attributes[7].values {
    Some(Strings(values)) => {
      assert_eq(values.length(), 1)
      assert_eq(values[0], value)
    }
    _ => abort("expected continued fixed-string attribute")
  }
  let attribute_header = parse_object_header(
    bytes,
    file.resolve("/continued").address,
    8,
    8,
    0UL,
  )
  assert_true(
    attribute_header.messages.any(message => {
      message.msg_type == msg_object_continuation
    }),
  )

  let links_header = parse_object_header(
    bytes,
    file.resolve("/continued_links").address,
    8,
    8,
    0UL,
  )
  let links = file.links(links_header)
  assert_eq(links.length(), 8)
  assert_eq(links[7].name, "link_7")
  assert_eq(links[7].link_target, Some(target))
  assert_true(
    links_header.messages.any(message => {
      message.msg_type == msg_object_continuation
    }),
  )
}

///|
test "writer chains object header continuations" {
  let writer = Hdf5Writer::new()
  let value = "m".repeat(60000)
  writer.add_dataset("/multi_continuation", [], UInt64Values([1UL]))
  for index in 0..<20 {
    writer.add_attribute(
      "/multi_continuation",
      "multi_" + index.to_string(),
      [],
      FixedStringValues([value], 60000),
    )
  }

  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  let attributes = file.attributes("/multi_continuation")
  assert_eq(attributes.length(), 20)
  match attributes[19].values {
    Some(Strings(values)) => assert_eq(values, [value])
    _ => abort("expected last chained continuation attribute")
  }
  let header = parse_object_header(
    bytes,
    file.resolve("/multi_continuation").address,
    8,
    8,
    0UL,
  )
  assert_eq(header.find(msg_object_continuation).length(), 2)
}

///|
test "writer round-trips absolute relative and dangling soft links" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/experiment/counts", [3UL], Int64Values([1L, 2L, 3L]))
  writer.add_group("/experiment/archive")
  writer.add_soft_link("/experiment", "latest", "/experiment/counts")
  writer.add_soft_link("/experiment", "relative", "counts")
  writer.add_soft_link("/experiment", "archive_link", "archive")
  writer.add_soft_link("/experiment", "missing", "/does/not/exist")

  let file = Hdf5File::parse(writer.encode())
  assert_eq(file.resolve("/experiment/latest").path, "/experiment/counts")
  assert_eq(file.resolve("/experiment/latest").kind, Dataset)
  assert_eq(integer_values(file.read_dataset("/experiment/latest")), [
    1L, 2L, 3L,
  ])
  assert_eq(file.resolve("/experiment/relative").path, "/experiment/counts")
  assert_eq(
    file.resolve("/experiment/archive_link").path,
    "/experiment/archive",
  )
  assert_eq(file.resolve("/experiment/archive_link").kind, Group)
  let dangling = try {
    file.resolve("/experiment/missing") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(dangling, Some(InvalidField("path component not found: does")))
}

///|
test "writer round-trips external links" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/experiment/counts", [3UL], Int64Values([1L, 2L, 3L]))
  writer.add_external_link(
    "/experiment", "external", "external-target.h5", "/experiment/counts",
  )
  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  let header = file.object_header_at(file.resolve("/experiment").address)
  let links = file.links(header)
  let index = links.search_by(link => link.name == "external").unwrap()
  assert_eq(links[index].cache_type, 64U)
  match links[index].link_target {
    Some(target) => assert_eq(target, "external-target.h5")
    None => abort("expected external link target file")
  }
  match links[index].external_path {
    Some(path) => assert_eq(path, "/experiment/counts")
    None => abort("expected external link target path")
  }
  assert_true(
    file
    .walk()
    .any(node => node.path == "/experiment/external" && node.kind == Unknown),
  )
  let unresolved = try {
    file.resolve("/experiment/external") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    unresolved,
    Some(
      UnsupportedFeature(
        "external links are not resolved: external-target.h5:/experiment/counts",
      ),
    ),
  )
}

///|
test "writer rejects invalid and conflicting external links" {
  let writer = Hdf5Writer::new()
  let empty_name = try {
    writer.add_external_link("/", "", "target.h5", "/target")
    None
  } catch {
    error => Some(error)
  }
  let slash_name = try {
    writer.add_external_link("/", "bad/name", "target.h5", "/target")
    None
  } catch {
    error => Some(error)
  }
  let empty_file = try {
    writer.add_external_link("/", "link", "", "/target")
    None
  } catch {
    error => Some(error)
  }
  let nul_file = try {
    writer.add_external_link("/", "link", "bad\u{0}file", "/target")
    None
  } catch {
    error => Some(error)
  }
  let empty_path = try {
    writer.add_external_link("/", "link", "target.h5", "")
    None
  } catch {
    error => Some(error)
  }
  let nul_path = try {
    writer.add_external_link("/", "link", "target.h5", "/bad\u{0}path")
    None
  } catch {
    error => Some(error)
  }
  let too_long = try {
    writer.add_external_link("/", "link", "target.h5", "p".repeat(65521))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(empty_name, Some(InvalidArgument("external link name is empty")))
  assert_eq(
    slash_name,
    Some(InvalidArgument("external link names cannot contain '/'")),
  )
  assert_eq(
    empty_file,
    Some(InvalidArgument("external link target file is empty")),
  )
  assert_eq(
    nul_file,
    Some(InvalidArgument("external link target file cannot contain NUL")),
  )
  assert_eq(
    empty_path,
    Some(InvalidArgument("external link target path is empty")),
  )
  assert_eq(
    nul_path,
    Some(InvalidArgument("external link target path cannot contain NUL")),
  )
  assert_eq(
    too_long,
    Some(InvalidArgument("external link message is too long")),
  )

  writer.add_dataset("/real", [], Int64Values([1L]))
  let occupied = try {
    writer.add_external_link("/", "real", "target.h5", "/target")
    None
  } catch {
    error => Some(error)
  }
  writer.add_external_link("/", "link", "target.h5", "/target")
  let duplicate = try {
    writer.add_external_link("/", "link", "other.h5", "/other")
    None
  } catch {
    error => Some(error)
  }
  let group_conflict = try {
    writer.add_group("/link/child")
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    occupied,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    duplicate,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    group_conflict,
    Some(InvalidArgument("writer path conflicts with an external link")),
  )
}

///|
test "writer rejects invalid and conflicting soft links" {
  let writer = Hdf5Writer::new()
  let empty_name = try {
    writer.add_soft_link("/", "", "/target")
    None
  } catch {
    error => Some(error)
  }
  let slash_name = try {
    writer.add_soft_link("/", "bad/name", "/target")
    None
  } catch {
    error => Some(error)
  }
  let dot_name = try {
    writer.add_soft_link("/", ".", "/target")
    None
  } catch {
    error => Some(error)
  }
  let nul_name = try {
    writer.add_soft_link("/", "bad\u{0}name", "/target")
    None
  } catch {
    error => Some(error)
  }
  let empty_target = try {
    writer.add_soft_link("/", "link", "")
    None
  } catch {
    error => Some(error)
  }
  let nul_target = try {
    writer.add_soft_link("/", "link", "/bad\u{0}target")
    None
  } catch {
    error => Some(error)
  }
  let too_long_target = try {
    writer.add_soft_link("/", "link", "t".repeat(65524))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(empty_name, Some(InvalidArgument("soft link name is empty")))
  assert_eq(
    slash_name,
    Some(InvalidArgument("soft link names cannot contain '/'")),
  )
  assert_eq(
    dot_name,
    Some(InvalidArgument("soft link names cannot be dot segments")),
  )
  assert_eq(
    nul_name,
    Some(InvalidArgument("soft link names cannot contain NUL")),
  )
  assert_eq(empty_target, Some(InvalidArgument("soft link target is empty")))
  assert_eq(
    nul_target,
    Some(InvalidArgument("soft link targets cannot contain NUL")),
  )
  assert_eq(
    too_long_target,
    Some(InvalidArgument("soft link message is too long")),
  )

  writer.add_dataset("/real", [], Int64Values([1L]))
  let occupied = try {
    writer.add_soft_link("/", "real", "/other")
    None
  } catch {
    error => Some(error)
  }
  writer.add_soft_link("/", "link", "/real")
  let duplicate = try {
    writer.add_soft_link("/", "link", "/real")
    None
  } catch {
    error => Some(error)
  }
  let dataset_conflict = try {
    writer.add_dataset("/link", [], Int64Values([2L]))
    None
  } catch {
    error => Some(error)
  }
  let group_conflict = try {
    writer.add_group("/link/child")
    None
  } catch {
    error => Some(error)
  }
  let parent_conflict = try {
    writer.add_soft_link("/link", "child", "/real")
    None
  } catch {
    error => Some(error)
  }
  let dataset_parent = try {
    writer.add_soft_link("/real", "child", "/real")
    None
  } catch {
    error => Some(error)
  }
  let attribute_target = try {
    writer.add_attribute("/link", "unit", [], UInt64Values([1UL]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    occupied,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    duplicate,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    dataset_conflict,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    group_conflict,
    Some(InvalidArgument("writer path conflicts with a soft link")),
  )
  assert_eq(
    parent_conflict,
    Some(InvalidArgument("writer path conflicts with a soft link")),
  )
  assert_eq(
    dataset_parent,
    Some(InvalidArgument("writer path conflicts with a dataset")),
  )
  assert_eq(
    attribute_target,
    Some(InvalidArgument("attribute target does not exist")),
  )
}

///|
test "writer stores large metadata in dense storage" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/many_attributes", [], UInt64Values([0UL]))
  for index in 0..<2500 {
    writer.add_attribute(
      "/many_attributes",
      "attr_" + index.to_string(),
      [],
      UInt64Values([index.to_uint64()]),
    )
  }
  writer.add_group("/many_links")
  for index in 0..<2500 {
    writer.add_soft_link(
      "/many_links",
      "link_" + index.to_string(),
      "/many_attributes",
    )
  }
  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  let attributes = file.attributes("/many_attributes")
  assert_eq(attributes.length(), 2500)
  match attributes[0].values {
    Some(UnsignedIntegers(values)) => assert_eq(values, [0UL])
    _ => abort("expected first dense attribute")
  }
  match attributes[2499].values {
    Some(UnsignedIntegers(values)) => assert_eq(values, [2499UL])
    _ => abort("expected last dense attribute")
  }
  let header = parse_object_header(
    bytes,
    file.resolve("/many_attributes").address,
    8,
    8,
    0UL,
  )
  let attribute_info = header.find(msg_attribute_info)[0]
  let attribute_reader = Reader::new(attribute_info.data)
  attribute_reader.skip(2)
  assert_true(attribute_reader.read_uint_le(8) != 0xFFFFFFFFFFFFFFFFUL)
  assert_true(attribute_reader.read_uint_le(8) != 0xFFFFFFFFFFFFFFFFUL)
  assert_true(
    !header.messages.any(message => message.msg_type == msg_attribute),
  )
  assert_eq(file.resolve("/many_links/link_2499").path, "/many_attributes")
  let links_header = parse_object_header(
    bytes,
    file.resolve("/many_links").address,
    8,
    8,
    0UL,
  )
  let link_info = links_header.find(msg_link_info)[0]
  let link_reader = Reader::new(link_info.data)
  link_reader.skip(2)
  assert_true(link_reader.read_uint_le(8) != 0xFFFFFFFFFFFFFFFFUL)
  assert_true(link_reader.read_uint_le(8) != 0xFFFFFFFFFFFFFFFFUL)
  assert_true(
    !links_header.messages.any(message => message.msg_type == msg_link),
  )
}

///|
test "writer rejects invalid paths shapes strings and duplicates" {
  let writer = Hdf5Writer::new()
  let relative = try {
    writer.add_group("relative")
    None
  } catch {
    error => Some(error)
  }
  assert_eq(relative, Some(InvalidArgument("writer paths must be absolute")))

  let shape = try {
    writer.add_dataset("/bad", [2UL], Int64Values([1L]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    shape,
    Some(InvalidArgument("writer value count does not match dataspace")),
  )

  let text = try {
    writer.add_dataset("/bad", [1UL], FixedStringValues(["long"], 3))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(text, Some(InvalidArgument("fixed string value exceeds its width")))

  let dimension = try {
    writer.add_dataset("/bad", [0UL, 0x80000000UL], UInt64Values([]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(dimension, Some(InvalidArgument("writer dimension is too large")))

  writer.add_dataset("/ok", [], Int64Values([1L]))
  let duplicate = try {
    writer.add_dataset("/ok", [], Int64Values([2L]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    duplicate,
    Some(InvalidArgument("writer object path already exists")),
  )
}

///|
test "writer snapshots caller-owned value and dimension arrays" {
  let writer = Hdf5Writer::new()
  let dimensions = [2UL]
  let values = [10L, 20L]
  writer.add_dataset("/values", dimensions, Int64Values(values))
  dimensions[0] = 1UL
  values[0] = 99L
  let file = Hdf5File::parse(writer.encode())
  let dataset = file.read_dataset("/values")
  assert_eq(dataset.dataspace.dimensions, [2UL])
  assert_eq(integer_values(dataset), [10L, 20L])
}

///|
test "writer emits multi-chunk fills and filter pipelines" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/raw",
    [3UL, 5UL],
    [2U, 3U],
    UInt64Values([
      0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 10UL, 11UL, 12UL, 13UL, 14UL,
    ]),
  )
  let sparse = Array::make(35, -7L)
  sparse[0] = 42L
  sparse[17] = 99L
  sparse[34] = -2L
  writer.add_chunked_dataset(
    "/filtered",
    [5UL, 7UL],
    [2U, 3U],
    Int64Values(sparse),
    fill_value=Int64Values([-7L]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  let file = Hdf5File::parse(writer.encode())
  let raw = file.read_dataset("/raw")
  assert_true(raw.layout.chunk_index is FixedArray(_))
  assert_eq(unsigned_values(raw), [
    0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 10UL, 11UL, 12UL, 13UL, 14UL,
  ])
  let filtered = file.read_dataset("/filtered")
  assert_true(filtered.layout.chunk_index is FixedArray(_))
  assert_eq(integer_values(filtered), sparse)
  let selected = file.read_dataset_selection(
    "/filtered",
    Points([[4UL, 6UL], [0UL, 0UL], [2UL, 3UL], [1UL, 1UL]]),
  )
  match selected.values {
    Integers(values) => assert_eq(values, [-2L, 42L, 99L, -7L])
    _ => abort("expected filtered integer selection")
  }
}

///|
test "writer emits integer scale-offset chunks" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/auto",
    [8UL],
    [4U],
    Int32Values([10, -7, 12, 13, 100, -7, 101, 102]),
    fill_value=Int32Values([-7]),
    filters=[ScaleOffset(0)],
  )
  writer.add_chunked_dataset(
    "/fixed",
    [6UL],
    [3U],
    Int16Values([100, 101, 102, 200, 201, 202]),
    fill_value=Int16Values([-1]),
    filters=[ScaleOffset(4)],
  )
  writer.add_chunked_dataset(
    "/pipeline",
    [6UL],
    [3U],
    Int32Values([1000, 1001, 1002, 2000, 2001, 2002]),
    fill_value=Int32Values([-1]),
    filters=[ScaleOffset(0), Deflate(6), Fletcher32],
  )
  writer.add_chunked_dataset(
    "/full",
    [4UL],
    [2U],
    Int32Values([-5, -4, -3, -2]),
    fill_value=Int32Values([0]),
    filters=[ScaleOffset(32)],
  )
  writer.add_chunked_dataset(
    "/i8",
    [4UL],
    [4U],
    Int8Values([-5, -1, 7, 9]),
    fill_value=Int8Values([-1]),
    filters=[ScaleOffset(0)],
  )
  writer.add_chunked_dataset(
    "/u16",
    [3UL],
    [3U],
    UInt16Values([10000, 10001, 10002]),
    fill_value=UInt16Values([42]),
    filters=[ScaleOffset(0)],
  )
  writer.add_chunked_dataset(
    "/i64",
    [3UL],
    [3U],
    Int64Values([-(1L << 40), 0L, 1L << 40]),
    fill_value=Int64Values([-1L]),
    filters=[ScaleOffset(0)],
  )
  writer.add_chunked_dataset(
    "/u64",
    [3UL],
    [3U],
    UInt64Values([1UL << 40, (1UL << 40) + 1UL, (1UL << 40) + 2UL]),
    fill_value=UInt64Values([7UL]),
    filters=[ScaleOffset(0)],
  )
  let file = Hdf5File::parse(writer.encode())
  assert_eq(integer_values(file.read_dataset("/auto")), [
    10L, -7L, 12L, 13L, 100L, -7L, 101L, 102L,
  ])
  assert_eq(integer_values(file.read_dataset("/fixed")), [
    100L, 101L, 102L, 200L, 201L, 202L,
  ])
  assert_eq(integer_values(file.read_dataset("/pipeline")), [
    1000L, 1001L, 1002L, 2000L, 2001L, 2002L,
  ])
  assert_eq(integer_values(file.read_dataset("/full")), [-5L, -4L, -3L, -2L])
  assert_eq(integer_values(file.read_dataset("/i8")), [-5L, -1L, 7L, 9L])
  assert_eq(unsigned_values(file.read_dataset("/u16")), [
    10000UL, 10001UL, 10002UL,
  ])
  assert_eq(integer_values(file.read_dataset("/i64")), [
    -(1L << 40),
    0L,
    1L << 40,
  ])
  assert_eq(unsigned_values(file.read_dataset("/u64")), [
    1UL << 40,
    (1UL << 40) + 1UL,
    (1UL << 40) + 2UL,
  ])
}

///|
test "writer emits floating-point scale-offset chunks" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/f32",
    [6UL],
    [3U],
    Float32Values([
      Float(1.25),
      Float(2.5),
      Float(-1.75),
      Float(9.5),
      Float(-9.5),
      Float(0.0),
    ]),
    fill_value=Float32Values([Float(-9.5)]),
    filters=[ScaleOffset(2)],
  )
  writer.add_chunked_dataset(
    "/f64",
    [5UL],
    [5U],
    Float64Values([1.5, -2.5, 3.5, -9.5, 0.0]),
    fill_value=Float64Values([-9.5]),
    filters=[ScaleOffset(1)],
  )
  writer.add_chunked_dataset(
    "/pipeline",
    [6UL],
    [3U],
    Float32Values([
      Float(1.25),
      Float(2.5),
      Float(-1.75),
      Float(9.5),
      Float(-9.5),
      Float(0.0),
    ]),
    fill_value=Float32Values([Float(-9.5)]),
    filters=[ScaleOffset(2), Deflate(6)],
  )
  let file = Hdf5File::parse(writer.encode())
  assert_eq(float_values(file.read_dataset("/f32")), [
    1.25, 2.5, -1.75, 9.5, -9.5, 0.0,
  ])
  assert_eq(float_values(file.read_dataset("/f64")), [1.5, -2.5, 3.5, -9.5, 0.0])
  assert_eq(float_values(file.read_dataset("/pipeline")), [
    1.25, 2.5, -1.75, 9.5, -9.5, 0.0,
  ])
  let selected = file.read_dataset_selection(
    "/f32",
    Points([[0UL], [2UL], [4UL]]),
  )
  match selected.values {
    Floats(values) => assert_eq(values, [1.25, -1.75, -9.5])
    _ => abort("expected filtered floating-point selection")
  }
}

///|
test "writer validates scale-offset filters" {
  let writer = Hdf5Writer::new()
  let cases : Array[Hdf5Error?] = [
    try {
      writer.add_chunked_dataset(
        "/float",
        [1UL],
        [1U],
        Float32Values([Float(1.5)]),
        filters=[ScaleOffset(39)],
      )
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset("/bits", [1UL], [1U], Int32Values([1]), filters=[
        ScaleOffset(33),
      ])
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset("/duplicate", [1UL], [1U], Int32Values([1]), filters=[
        ScaleOffset(0),
        ScaleOffset(0),
      ])
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset("/order", [1UL], [1U], Int32Values([1]), filters=[
        Deflate(6),
        ScaleOffset(0),
      ])
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset(
        "/lossy_checksum",
        [1UL],
        [1U],
        Float32Values([Float(1.5)]),
        filters=[ScaleOffset(2), Fletcher32],
      )
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset("/full_range", [1UL], [1U], Int32Values([0]), filters=[
        ScaleOffset(0),
      ])
      writer.add_chunked_dataset("/wide", [1UL], [1U], UInt32Values([0U]), filters=[
        ScaleOffset(0),
      ])
      None
    } catch {
      error => Some(error)
    },
  ]
  assert_eq(cases, [
    Some(
      InvalidArgument(
        "scale-offset decimal digits exceed the datatype precision",
      ),
    ),
    Some(InvalidArgument("scale-offset minimum bits exceed the datatype width")),
    Some(InvalidArgument("scale-offset must be the first writer filter")),
    Some(InvalidArgument("scale-offset must be the first writer filter")),
    Some(
      InvalidArgument(
        "Fletcher32 cannot be combined with floating-point scale-offset",
      ),
    ),
    None,
  ])
}

///|
test "writer rejects automatic full-range scale-offset chunks" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/values",
    [2UL],
    [2U],
    Int8Values([-128, 127]),
    fill_value=Int8Values([0]),
    filters=[ScaleOffset(0)],
  )
  let result = try {
    writer.encode() |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    result,
    Some(
      UnsupportedFeature("scale-offset automatic bits require full precision"),
    ),
  )
}

///|
test "writer validates chunk metadata filters and fill types" {
  let writer = Hdf5Writer::new()
  let cases : Array[Hdf5Error?] = [
    try {
      writer.add_chunked_dataset(
        "/rank",
        [2UL],
        [1U, 1U],
        Int64Values([1L, 2L]),
      )
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset("/empty", [0UL], [1U], Int64Values([]))
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset(
        "/fill",
        [1UL],
        [1U],
        Int64Values([1L]),
        fill_value=UInt64Values([0UL]),
      )
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset("/filters", [1UL], [1U], Int64Values([1L]), filters=[
        Deflate(6),
        Shuffle,
      ])
      None
    } catch {
      error => Some(error)
    },
  ]
  assert_eq(cases, [
    Some(InvalidArgument("chunk dimensions must match a simple dataspace")),
    Some(InvalidArgument("chunk dimension is invalid")),
    Some(InvalidArgument("fill value must be one value of the dataset type")),
    Some(InvalidArgument("shuffle must be the first writer filter")),
  ])
}

///|
test "writer fixed arrays handle the page boundary" {
  let writer = Hdf5Writer::new()
  for (path, count) in [("/exact", 1024), ("/many", 1025)] {
    let values = Array::make(count, 0L)
    values[0] = 7L
    values[count - 1] = 9L
    writer.add_chunked_dataset(
      path,
      [count.to_uint64()],
      [1U],
      Int64Values(values),
    )
  }
  let file = Hdf5File::parse(writer.encode())
  for (path, count) in [("/exact", 1024), ("/many", 1025)] {
    let selected = file.read_dataset_selection(
      path,
      Points([[count.to_uint64() - 1UL], [512UL], [0UL]]),
    )
    match selected.values {
      Integers(items) => assert_eq(items, [9L, 0L, 7L])
      _ => fail("expected fixed-array values across the page boundary")
    }
  }
}

///|
test "writer appends resizes and fills extensible datasets" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/series",
    [0UL, 3UL],
    [None, Some(4UL)],
    [2U, 2U],
    Int64Values([]),
    fill_value=Int64Values([-1L]),
  )
  writer.append_dataset(
    "/series",
    [2UL, 3UL],
    Int64Values([1L, 2L, 3L, 4L, 5L, 6L]),
  )
  writer.append_dataset("/series", [1UL, 3UL], Int64Values([7L, 8L, 9L]))
  writer.resize_dataset("/series", [5UL, 3UL])
  let grown_file = Hdf5File::parse(writer.encode())
  let grown = grown_file.read_dataset("/series")
  assert_eq(grown.dataspace.dimensions, [5UL, 3UL])
  assert_eq(grown.dataspace.maximum_dimensions, [0xFFFFFFFFFFFFFFFFUL, 4UL])
  assert_true(grown.layout.chunk_index is ExtensibleArray(_))
  assert_eq(integer_values(grown), [
    1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, -1L, -1L, -1L, -1L, -1L, -1L,
  ])

  writer.resize_dataset("/series", [2UL, 3UL])
  let shrunk = Hdf5File::parse(writer.encode()).read_dataset("/series")
  assert_eq(integer_values(shrunk), [1L, 2L, 3L, 4L, 5L, 6L])
}

///|
test "writer extensible arrays use direct data blocks" {
  let values = Array::make(300, 0L)
  values[0] = 1L
  values[127] = 2L
  values[128] = 3L
  values[299] = 4L
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/many",
    [300UL],
    [None],
    [1U],
    Int64Values(values),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  let file = Hdf5File::parse(writer.encode())
  let selected = file.read_dataset_selection(
    "/many",
    Points([[299UL], [128UL], [127UL], [1UL], [0UL]]),
  )
  match selected.values {
    Integers(items) => assert_eq(items, [4L, 3L, 2L, 0L, 1L])
    _ => fail("expected extensible-array direct block values")
  }
}

///|
test "writer extensible arrays use super blocks and paged data blocks" {
  let values = Array::make(134200, 0U)
  values[0] = 1U
  values[243] = 2U
  values[244] = 3U
  values[65523] = 8U
  values[65524] = 9U
  values[131059] = 4U
  values[131060] = 5U
  values[132100] = 6U
  values[134199] = 7U
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/paged",
    [134200UL],
    [None],
    [1U],
    UInt8Values(values),
    filters=[Deflate(6), Fletcher32],
  )
  let file = Hdf5File::parse(writer.encode())
  let selected = file.read_dataset_selection(
    "/paged",
    Points([
      [134199UL],
      [132100UL],
      [131061UL],
      [131060UL],
      [131059UL],
      [65524UL],
      [65523UL],
      [244UL],
      [243UL],
      [0UL],
    ]),
  )
  match selected.values {
    UnsignedIntegers(items) =>
      assert_eq(items, [7UL, 6UL, 0UL, 5UL, 4UL, 9UL, 8UL, 3UL, 2UL, 1UL])
    _ => fail("expected extensible-array super-block values")
  }
}

///|
test "writer rejects chunk grids beyond supported capacity" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/large",
    [0UL, 50000UL],
    [None, Some(50000UL)],
    [1U, 1U],
    Int64Values([]),
  )
  let error = try {
    writer.resize_dataset("/large", [50000UL, 50000UL])
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidArgument("writer chunk grid is too large")))
}

///|
test "writer extensible indexes swizzle a nonleading unlimited dimension" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/filtered",
    [2UL, 0UL],
    [Some(2UL), None],
    [1U, 2U],
    UInt64Values([]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  writer.append_dataset(
    "/filtered",
    [2UL, 3UL],
    UInt64Values([1UL, 2UL, 3UL, 4UL, 5UL, 6UL]),
  )
  writer.append_dataset("/filtered", [2UL, 1UL], UInt64Values([7UL, 8UL]))
  let dataset = Hdf5File::parse(writer.encode()).read_dataset("/filtered")
  assert_eq(dataset.dataspace.dimensions, [2UL, 4UL])
  assert_eq(unsigned_values(dataset), [1UL, 2UL, 3UL, 7UL, 4UL, 5UL, 6UL, 8UL])
}

///|
test "writer validates extensible shapes and mutations" {
  let writer = Hdf5Writer::new()
  let missing_unlimited = try {
    writer.add_extensible_dataset(
      "/fixed",
      [2UL],
      [Some(4UL)],
      [2U],
      Int64Values([1L, 2L]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    missing_unlimited,
    Some(InvalidArgument("extensible dataset requires one unlimited dimension")),
  )
  let oversized_chunk = try {
    writer.add_extensible_dataset(
      "/oversized_chunk",
      [0UL, 1UL],
      [None, Some(1UL)],
      [0x7FFFFFFFU, 2U],
      Int64Values([]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(oversized_chunk, Some(InvalidArgument("writer chunk is too large")))
  writer.add_extensible_dataset(
    "/values",
    [1UL, 2UL],
    [None, Some(2UL)],
    [1U, 2U],
    Int64Values([1L, 2L]),
  )
  let maximum = try {
    writer.resize_dataset("/values", [2UL, 3UL])
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    maximum,
    Some(InvalidArgument("resized dataset exceeds its maximum dimensions")),
  )
  let append_shape = try {
    writer.append_dataset("/values", [1UL, 1UL], Int64Values([3L]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    append_shape,
    Some(InvalidArgument("appended block must match fixed dataset dimensions")),
  )
}

///|
test "writer round-trips narrow integers and float32 across layouts" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/i8", [3UL], Int8Values([-128, 0, 127]))
  writer.add_dataset("/i16", [3UL], Int16Values([-32768, 0, 32767]))
  writer.add_dataset("/i32", [3UL], Int32Values([-2147483647, 0, 2147483647]))
  writer.add_dataset("/u8", [3UL], UInt8Values([0U, 1U, 255U]))
  writer.add_dataset("/u16", [3UL], UInt16Values([0U, 1U, 65535U]))
  writer.add_dataset("/u32", [3UL], UInt32Values([0U, 1U, 0xFFFFFFFFU]))
  writer.add_dataset(
    "/f32",
    [3UL],
    Float32Values([Float(-1.5), Float(0.0), Float(2.25)]),
  )
  writer.add_chunked_dataset(
    "/chunked_i16",
    [5UL],
    [2U],
    Int16Values([-7, 2, -7, 4, -7]),
    fill_value=Int16Values([-7]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  writer.add_extensible_dataset(
    "/growing_f32",
    [0UL],
    [None],
    [2U],
    Float32Values([]),
    fill_value=Float32Values([Float(-1.0)]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  writer.append_dataset(
    "/growing_f32",
    [3UL],
    Float32Values([Float(1.25), Float(2.5), Float(3.75)]),
  )
  writer.resize_dataset("/growing_f32", [4UL])

  let file = Hdf5File::parse(writer.encode())
  assert_eq(integer_values(file.read_dataset("/i8")), [-128L, 0L, 127L])
  assert_eq(integer_values(file.read_dataset("/i16")), [-32768L, 0L, 32767L])
  assert_eq(integer_values(file.read_dataset("/i32")), [
    -2147483647L, 0L, 2147483647L,
  ])
  assert_eq(unsigned_values(file.read_dataset("/u8")), [0UL, 1UL, 255UL])
  assert_eq(unsigned_values(file.read_dataset("/u16")), [0UL, 1UL, 65535UL])
  assert_eq(unsigned_values(file.read_dataset("/u32")), [0UL, 1UL, 0xFFFFFFFFUL])
  assert_eq(float_values(file.read_dataset("/f32")), [-1.5, 0.0, 2.25])
  assert_eq(integer_values(file.read_dataset("/chunked_i16")), [
    -7L, 2L, -7L, 4L, -7L,
  ])
  assert_eq(float_values(file.read_dataset("/growing_f32")), [
    1.25, 2.5, 3.75, -1.0,
  ])
}

///|
test "writer rejects narrow integer values outside their datatype" {
  let writer = Hdf5Writer::new()
  let signed = try {
    writer.add_dataset("/signed", [1UL], Int8Values([128]))
    None
  } catch {
    error => Some(error)
  }
  let unsigned = try {
    writer.add_dataset("/unsigned", [1UL], UInt16Values([65536U]))
    None
  } catch {
    error => Some(error)
  }
  let fill = try {
    writer.add_chunked_dataset(
      "/fill",
      [1UL],
      [1U],
      Int8Values([1]),
      fill_value=Int8Values([-129]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(signed, Some(InvalidArgument("Int8 writer value is out of range")))
  assert_eq(
    unsigned,
    Some(InvalidArgument("UInt16 writer value is out of range")),
  )
  assert_eq(fill, Some(InvalidArgument("Int8 writer value is out of range")))
}

///|
fn writer_record_type() -> Datatype raise Hdf5Error {
  let status = Datatype::write_enum(Datatype::write_integer(false, 1), [
    { name: "MISSING", value: EnumValue::Unsigned(0UL), },
    { name: "READY", value: EnumValue::Unsigned(1UL), },
    { name: "DONE", value: EnumValue::Unsigned(2UL), },
  ])
  let samples = Datatype::write_array(Datatype::write_integer(true, 2), [
    2UL, 2UL,
  ])
  Datatype::write_compound([
    ("id", Datatype::write_integer(true, 4)),
    ("status", status),
    ("samples", samples),
    ("label", Datatype::write_fixed_string(6)),
  ])
}

///|
fn writer_record(
  id : Int64,
  status : String,
  samples : Array[Int64],
  label : String,
) -> DatasetValue {
  Compound([
    { name: "id", value: SignedInteger(id), },
    { name: "status", value: Enumeration(status), },
    {
      name: "samples",
      value: Elements(samples.map(value => SignedInteger(value))),
    },
    { name: "label", value: Text(label), },
  ])
}

///|
test "writer encodes compound enum and array values recursively" {
  let writer = Hdf5Writer::new()
  let record_type = writer_record_type()
  let missing = writer_record(0L, "MISSING", [0L, 0L, 0L, 0L], "")
  let first = writer_record(7L, "READY", [1L, 2L, 3L, 4L], "alpha")
  let second = writer_record(-2L, "DONE", [-1L, -2L, -3L, -4L], "beta")
  writer.add_chunked_dataset(
    "/records",
    [3UL],
    [2U],
    StructuredValues(record_type, [first, missing, second]),
    fill_value=StructuredValues(record_type, [missing]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  let array_type = Datatype::write_array(Datatype::write_float(4), [2UL])
  writer.add_dataset(
    "/vectors",
    [2UL],
    StructuredValues(array_type, [
      Elements([FloatingPoint(1.5), FloatingPoint(-2.0)]),
      Elements([FloatingPoint(3.25), FloatingPoint(4.5)]),
    ]),
  )
  writer.add_attribute(
    "/records",
    "prototype",
    [],
    StructuredValues(record_type, [first]),
  )

  let file = Hdf5File::parse(writer.encode())
  let records = file.read_dataset("/records")
  assert_eq(records.datatype.class, 6)
  assert_eq(records.datatype.compound_members.length(), 4)
  assert_eq(structured_values(records), [first, missing, second])
  assert_eq(structured_values(file.read_dataset("/vectors")), [
    Elements([FloatingPoint(1.5), FloatingPoint(-2.0)]),
    Elements([FloatingPoint(3.25), FloatingPoint(4.5)]),
  ])
  match file.attributes("/records")[0].values {
    Some(Structured(values)) => assert_eq(values, [first])
    _ => abort("expected compound attribute")
  }
}

///|
test "writer round-trips nonstandard integer widths" {
  let writer = Hdf5Writer::new()
  let signed_24 = Datatype::write_integer(true, 3)
  let unsigned_40 = Datatype::write_integer(false, 5)
  writer.add_dataset(
    "/signed_24",
    [3UL],
    StructuredValues(signed_24, [
      SignedInteger(42L),
      SignedInteger(-2L),
      SignedInteger(-8388608L),
    ]),
  )
  writer.add_dataset(
    "/unsigned_40",
    [3UL],
    StructuredValues(unsigned_40, [
      UnsignedInteger(0UL),
      UnsignedInteger(1UL),
      UnsignedInteger(0xFFFFFFFFFFUL),
    ]),
  )

  let file = Hdf5File::parse(writer.encode())
  let signed = file.read_dataset("/signed_24")
  assert_eq(signed.datatype.size, 3)
  assert_eq(signed.datatype.bit_precision, 24)
  assert_eq(integer_values(signed), [42L, -2L, -8388608L])
  let unsigned = file.read_dataset("/unsigned_40")
  assert_eq(unsigned.datatype.size, 5)
  assert_eq(unsigned.datatype.bit_precision, 40)
  assert_eq(unsigned_values(unsigned), [0UL, 1UL, 0xFFFFFFFFFFUL])
}

///|
test "writer round-trips big-endian numeric enum and mixed compound values" {
  let writer = Hdf5Writer::new()
  let big_i16 = Datatype::write_integer_with_order(true, 2, false)
  let big_u64 = Datatype::write_integer_with_order(false, 8, false)
  let big_f32 = Datatype::write_float_with_order(4, false)
  let big_f64 = Datatype::write_float_with_order(8, false)
  writer.add_dataset(
    "/big_i16",
    [3UL],
    StructuredValues(big_i16, [
      SignedInteger(-32768L),
      SignedInteger(-1L),
      SignedInteger(12345L),
    ]),
  )
  writer.add_dataset(
    "/big_u64",
    [3UL],
    StructuredValues(big_u64, [
      UnsignedInteger(0UL),
      UnsignedInteger(1UL << 40),
      UnsignedInteger(0xFFFFFFFFFFFFFFFFUL),
    ]),
  )
  writer.add_dataset(
    "/big_f32",
    [3UL],
    StructuredValues(big_f32, [
      FloatingPoint(-1.5),
      FloatingPoint(0.0),
      FloatingPoint(2.25),
    ]),
  )
  writer.add_dataset(
    "/big_f64",
    [3UL],
    StructuredValues(big_f64, [
      FloatingPoint(-1.25),
      FloatingPoint(0.0),
      FloatingPoint(3.5),
    ]),
  )

  let state = Datatype::write_enum(
    Datatype::write_integer_with_order(true, 2, false),
    [
      { name: "LOW", value: EnumValue::Signed(-1L), },
      { name: "HIGH", value: EnumValue::Signed(300L), },
    ],
  )
  writer.add_dataset(
    "/big_state",
    [2UL],
    StructuredValues(state, [Enumeration("HIGH"), Enumeration("LOW")]),
  )

  let samples = Datatype::write_array(
    Datatype::write_integer_with_order(false, 4, false),
    [2UL],
  )
  let record_type = Datatype::write_compound([
    ("id", Datatype::write_integer(true, 4)),
    ("weight", big_f64),
    ("count", Datatype::write_integer_with_order(false, 2, false)),
    ("state", state),
    ("samples", samples),
  ])
  let record = Compound([
    { name: "id", value: SignedInteger(7L), },
    { name: "weight", value: FloatingPoint(-2.5), },
    { name: "count", value: UnsignedInteger(513UL), },
    { name: "state", value: Enumeration("HIGH"), },
    {
      name: "samples",
      value: Elements([UnsignedInteger(1UL), UnsignedInteger(0x10203040UL)]),
    },
  ])
  writer.add_dataset(
    "/big_record",
    [1UL],
    StructuredValues(record_type, [record]),
  )
  writer.add_attribute(
    "/big_record",
    "marker",
    [],
    StructuredValues(Datatype::write_integer_with_order(true, 4, false), [
      SignedInteger(-5L),
    ]),
  )
  writer.add_attribute(
    "/big_record",
    "prototype",
    [],
    StructuredValues(record_type, [record]),
  )

  writer.add_extensible_dataset(
    "/big_fill",
    [1UL],
    [None],
    [2U],
    StructuredValues(big_i16, [SignedInteger(7L)]),
    fill_value=StructuredValues(big_i16, [SignedInteger(-999L)]),
  )
  writer.resize_dataset("/big_fill", [4UL])

  let file = Hdf5File::parse(writer.encode())
  let integers = file.read_dataset("/big_i16")
  assert_eq(integers.datatype.little_endian, false)
  assert_eq(integer_values(integers), [-32768L, -1L, 12345L])
  let unsigned = file.read_dataset("/big_u64")
  assert_eq(unsigned.datatype.little_endian, false)
  assert_eq(unsigned_values(unsigned), [0UL, 1UL << 40, 0xFFFFFFFFFFFFFFFFUL])
  let floats32 = file.read_dataset("/big_f32")
  assert_eq(floats32.datatype.little_endian, false)
  assert_eq(float_values(floats32), [-1.5, 0.0, 2.25])
  let floats64 = file.read_dataset("/big_f64")
  assert_eq(floats64.datatype.little_endian, false)
  assert_eq(float_values(floats64), [-1.25, 0.0, 3.5])
  let states = file.read_dataset("/big_state")
  assert_eq(states.datatype.little_endian, false)
  assert_eq(states.datatype.base_types[0].little_endian, false)
  match states.values {
    Enums(values) => assert_eq(values, ["HIGH", "LOW"])
    _ => abort("expected big-endian enum values")
  }
  assert_eq(states.datatype.enum_members[0].value, EnumValue::Signed(-1L))
  assert_eq(states.datatype.enum_members[1].value, EnumValue::Signed(300L))

  let records = file.read_dataset("/big_record")
  let fields = records.datatype.compound_members
  assert_eq(fields[0].datatype.little_endian, true)
  assert_eq(fields[1].datatype.little_endian, false)
  assert_eq(fields[2].datatype.little_endian, false)
  assert_eq(fields[3].datatype.little_endian, false)
  assert_eq(fields[4].datatype.base_types[0].little_endian, false)
  assert_eq(structured_values(records), [record])
  let attributes = file.attributes("/big_record")
  match attributes.search_by(attribute => attribute.name == "marker") {
    Some(index) => {
      assert_eq(attributes[index].datatype.little_endian, false)
      match attributes[index].values {
        Some(Integers(values)) => assert_eq(values, [-5L])
        _ => abort("expected big-endian integer attribute")
      }
    }
    None => abort("expected big-endian integer attribute")
  }
  match attributes.search_by(attribute => attribute.name == "prototype") {
    Some(index) =>
      match attributes[index].values {
        Some(Structured(values)) => assert_eq(values, [record])
        _ => abort("expected big-endian compound attribute")
      }
    None => abort("expected big-endian compound attribute")
  }
  assert_eq(integer_values(file.read_dataset("/big_fill")), [
    7L, -999L, -999L, -999L,
  ])
}

///|
test "writer round-trips time bitfield and opaque values" {
  let writer = Hdf5Writer::new()
  let time = Datatype::write_time(8)
  let big_time = Datatype::write_time(4, little_endian=false)
  let flags = Datatype::write_bitfield(2, 12, offset=4)
  let opaque_type = Datatype::write_opaque(4, "MOON|V4")
  let untagged = Datatype::write_opaque(3, "")
  writer.add_dataset(
    "/time",
    [3UL],
    StructuredValues(time, [Time(0UL), Time(1700000000UL), Time(42UL)]),
  )
  writer.add_dataset(
    "/big_time",
    [2UL],
    StructuredValues(big_time, [Time(0UL), Time(0x01020304UL)]),
  )
  writer.add_dataset(
    "/flags",
    [3UL],
    StructuredValues(flags, [Bitfield(0UL), Bitfield(1UL), Bitfield(0xABCUL)]),
  )
  writer.add_chunked_dataset(
    "/opaque",
    [2UL],
    [1U],
    StructuredValues(opaque_type, [Opaque(b"\x00\x01\xfe\xff"), Opaque(b"MOON")]),
    filters=[Shuffle, Deflate(6)],
  )
  writer.add_dataset(
    "/untagged",
    [1UL],
    StructuredValues(untagged, [Opaque(b"abc")]),
  )

  let record_type = Datatype::write_compound([
    ("when", time),
    ("flags", flags),
    ("blob", opaque_type),
  ])
  let record = Compound([
    { name: "when", value: Time(1700000000UL), },
    { name: "flags", value: Bitfield(0x345UL), },
    { name: "blob", value: Opaque(b"DATA"), },
  ])
  writer.add_dataset("/record", [1UL], StructuredValues(record_type, [record]))
  writer.add_attribute(
    "/record",
    "flag_marker",
    [],
    StructuredValues(flags, [Bitfield(0x7FFUL)]),
  )
  writer.add_extensible_dataset(
    "/time_fill",
    [1UL],
    [None],
    [2U],
    StructuredValues(time, [Time(7UL)]),
    fill_value=StructuredValues(time, [Time(99UL)]),
  )
  writer.resize_dataset("/time_fill", [3UL])

  let file = Hdf5File::parse(writer.encode())
  let time_dataset = file.read_dataset("/time")
  assert_eq(time_dataset.datatype.class, 2)
  assert_eq(time_dataset.datatype.bit_precision, 64)
  assert_eq(structured_values(time_dataset), [
    Time(0UL),
    Time(1700000000UL),
    Time(42UL),
  ])
  let big_time_dataset = file.read_dataset("/big_time")
  assert_eq(big_time_dataset.datatype.little_endian, false)
  assert_eq(structured_values(big_time_dataset), [Time(0UL), Time(0x01020304UL)])
  let flag_dataset = file.read_dataset("/flags")
  assert_eq(flag_dataset.datatype.class, 4)
  assert_eq(flag_dataset.datatype.bit_offset, 4)
  assert_eq(flag_dataset.datatype.bit_precision, 12)
  assert_eq(structured_values(flag_dataset), [
    Bitfield(0UL),
    Bitfield(1UL),
    Bitfield(0xABCUL),
  ])
  let opaque_dataset = file.read_dataset("/opaque")
  assert_eq(opaque_dataset.datatype.class, 5)
  assert_eq(opaque_dataset.datatype.opaque_tag, "MOON|V4")
  assert_eq(structured_values(opaque_dataset), [
    Opaque(b"\x00\x01\xfe\xff"),
    Opaque(b"MOON"),
  ])
  assert_eq(file.read_dataset("/untagged").datatype.opaque_tag, "")
  assert_eq(structured_values(file.read_dataset("/record")), [record])
  let attributes = file.attributes("/record")
  match attributes.search_by(attribute => attribute.name == "flag_marker") {
    Some(index) =>
      match attributes[index].values {
        Some(Structured(values)) => assert_eq(values, [Bitfield(0x7FFUL)])
        _ => abort("expected bitfield attribute")
      }
    None => abort("expected bitfield attribute")
  }
  assert_eq(structured_values(file.read_dataset("/time_fill")), [
    Time(7UL),
    Time(99UL),
    Time(99UL),
  ])
}

///|
test "writer round-trips variable-length sequences" {
  let writer = Hdf5Writer::new()
  let sequence = Datatype::write_vlen_sequence(Datatype::write_integer(true, 4))
  writer.add_dataset(
    "/sequences",
    [4UL],
    StructuredValues(sequence, [
      Sequence([]),
      Sequence([SignedInteger(1L), SignedInteger(-2L), SignedInteger(3L)]),
      Sequence([SignedInteger(1L), SignedInteger(-2L), SignedInteger(3L)]),
      Sequence([SignedInteger(-1L), SignedInteger(0L)]),
    ]),
  )
  writer.add_attribute(
    "/sequences",
    "prototype",
    [],
    StructuredValues(sequence, [
      Sequence([SignedInteger(7L), SignedInteger(8L)]),
    ]),
  )

  let file = Hdf5File::parse(writer.encode())
  let dataset = file.read_dataset("/sequences")
  assert_eq(dataset.datatype.class, 9)
  assert_eq(dataset.datatype.vlen_is_string, false)
  assert_eq(dataset.datatype.base_types[0].size, 4)
  assert_eq(structured_values(dataset), [
    Sequence([]),
    Sequence([SignedInteger(1L), SignedInteger(-2L), SignedInteger(3L)]),
    Sequence([SignedInteger(1L), SignedInteger(-2L), SignedInteger(3L)]),
    Sequence([SignedInteger(-1L), SignedInteger(0L)]),
  ])
  let attributes = file.attributes("/sequences")
  match attributes[0].values {
    Some(Structured(values)) =>
      assert_eq(values, [Sequence([SignedInteger(7L), SignedInteger(8L)])])
    _ => abort("expected variable-length sequence attribute")
  }
}

///|
test "writer rejects invalid time bitfield opaque and sequence types" {
  let invalid_time : Hdf5Error? = try {
    Datatype::write_time(3) |> ignore
    None
  } catch {
    error => Some(error)
  }
  let invalid_bitfield : Hdf5Error? = try {
    Datatype::write_bitfield(2, 13, offset=4) |> ignore
    None
  } catch {
    error => Some(error)
  }
  let invalid_opaque : Hdf5Error? = try {
    Datatype::write_opaque(0, "") |> ignore
    None
  } catch {
    error => Some(error)
  }
  let nested_sequence : Hdf5Error? = try {
    let base = Datatype::write_vlen_sequence(Datatype::write_integer(true, 4))
    Datatype::write_vlen_sequence(base) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    invalid_time,
    Some(InvalidArgument("writer time datatype width is invalid")),
  )
  assert_eq(
    invalid_bitfield,
    Some(InvalidArgument("writer bitfield layout is invalid")),
  )
  assert_eq(
    invalid_opaque,
    Some(InvalidArgument("writer opaque datatype width is invalid")),
  )
  assert_eq(
    nested_sequence,
    Some(InvalidArgument("writer variable-length sequence base is invalid")),
  )

  let writer = Hdf5Writer::new()
  let sequence = Datatype::write_vlen_sequence(Datatype::write_integer(true, 4))
  let filtered = try {
    writer.add_chunked_dataset(
      "/sequences",
      [1UL],
      [1U],
      StructuredValues(sequence, [Sequence([SignedInteger(1L)])]),
      filters=[Deflate(6)],
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    filtered,
    Some(
      UnsupportedFeature(
        "filters are not supported for variable-length datasets",
      ),
    ),
  )
  let invalid_fill = try {
    writer.add_chunked_dataset(
      "/fill",
      [1UL],
      [1U],
      StructuredValues(sequence, [Sequence([])]),
      fill_value=StructuredValues(sequence, [Sequence([SignedInteger(1L)])]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    invalid_fill,
    Some(
      UnsupportedFeature(
        "custom variable-length sequence fills are not supported yet",
      ),
    ),
  )
}

///|
test "writer appends and fills extensible enums" {
  let writer = Hdf5Writer::new()
  let state = Datatype::write_enum(Datatype::write_integer(false, 1), [
    { name: "UNKNOWN", value: EnumValue::Unsigned(0UL), },
    { name: "OPEN", value: EnumValue::Unsigned(1UL), },
    { name: "CLOSED", value: EnumValue::Unsigned(2UL), },
  ])
  writer.add_extensible_dataset(
    "/states",
    [0UL],
    [None],
    [2U],
    StructuredValues(state, []),
    fill_value=StructuredValues(state, [Enumeration("UNKNOWN")]),
  )
  writer.append_dataset(
    "/states",
    [2UL],
    StructuredValues(state, [Enumeration("OPEN"), Enumeration("CLOSED")]),
  )
  writer.resize_dataset("/states", [4UL])
  match Hdf5File::parse(writer.encode()).read_dataset("/states").values {
    Enums(values) => assert_eq(values, ["OPEN", "CLOSED", "UNKNOWN", "UNKNOWN"])
    _ => abort("expected enum values")
  }
}

///|
test "writer validates structured values before encoding" {
  let writer = Hdf5Writer::new()
  let record_type = writer_record_type()
  let wrong_fields = try {
    writer.add_dataset(
      "/bad",
      [1UL],
      StructuredValues(record_type, [
        Compound([
          { name: "status", value: Enumeration("READY"), },
          { name: "id", value: SignedInteger(1L), },
          {
            name: "samples",
            value: Elements([
              SignedInteger(1L),
              SignedInteger(2L),
              SignedInteger(3L),
              SignedInteger(4L),
            ]),
          },
          { name: "label", value: Text("bad"), },
        ]),
      ]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_fields,
    Some(InvalidArgument("compound value fields are out of order")),
  )
  let oversized_enum = try {
    let invalid = Datatype::write_enum(Datatype::write_integer(false, 1), [
      { name: "TOO_LARGE", value: EnumValue::Unsigned(256UL), },
    ])
    writer.add_dataset(
      "/enum",
      [1UL],
      StructuredValues(invalid, [Enumeration("TOO_LARGE")]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    oversized_enum,
    Some(InvalidArgument("structured unsigned integer is out of range")),
  )
}

///|
test "writer round-trips variable-length UTF-8 strings" {
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/names",
    [4UL],
    VariableStringValues(["月亮", "HDF5", "", "月亮"]),
  )
  writer.add_chunked_dataset(
    "/chunked_names",
    [5UL],
    [2U],
    VariableStringValues(["alpha", "", "数据", "alpha", ""]),
  )
  writer.add_extensible_dataset(
    "/bounded_names",
    [1UL, 2UL],
    [None, Some(4UL)],
    [1U, 2U],
    VariableStringValues(["a", "b"]),
  )
  writer.add_extensible_dataset(
    "/events",
    [0UL],
    [None],
    [2U],
    VariableStringValues([]),
  )
  writer.append_dataset(
    "/events",
    [2UL],
    VariableStringValues(["start", "结束"]),
  )
  writer.resize_dataset("/events", [3UL])
  writer.add_attribute(
    "/names",
    "description",
    [],
    VariableStringValues(["UTF-8 标签"]),
  )

  let file = Hdf5File::parse(writer.encode())
  let names = file.read_dataset("/names")
  assert_eq(names.datatype.class, 9)
  assert_eq(names.datatype.version, 1)
  assert_eq(names.datatype.base_types.length(), 1)
  assert_eq(names.datatype.base_types[0].class, 0)
  assert_eq(names.datatype.base_types[0].size, 1)
  assert_eq(string_values(file.read_dataset("/names")), [
    "月亮", "HDF5", "", "月亮",
  ])
  assert_eq(string_values(file.read_dataset("/chunked_names")), [
    "alpha", "", "数据", "alpha", "",
  ])
  assert_eq(string_values(file.read_dataset("/bounded_names")), ["a", "b"])
  assert_eq(string_values(file.read_dataset("/events")), ["start", "结束", ""])
  match file.attributes("/names")[0].values {
    Some(Strings(values)) => assert_eq(values, ["UTF-8 标签"])
    _ => abort("expected variable-length string attribute")
  }
}

///|
test "writer rejects filters for variable-length datasets" {
  let chunked = try {
    Hdf5Writer::new().add_chunked_dataset(
      "/strings",
      [2UL],
      [1U],
      VariableStringValues(["alpha", ""]),
      filters=[Deflate(6), Fletcher32],
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    chunked,
    Some(
      UnsupportedFeature(
        "filters are not supported for variable-length datasets",
      ),
    ),
  )
  let extensible = try {
    Hdf5Writer::new().add_extensible_dataset(
      "/strings",
      [0UL],
      [None],
      [1U],
      VariableStringValues([]),
      filters=[Shuffle],
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    extensible,
    Some(
      UnsupportedFeature(
        "filters are not supported for variable-length datasets",
      ),
    ),
  )
}

///|
test "writer round-trips custom variable-length string fills" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/strings",
    [3UL],
    [2U],
    VariableStringValues(["value", "", ""]),
    fill_value=VariableStringValues(["missing"]),
  )
  writer.add_extensible_dataset(
    "/extensible_strings",
    [0UL],
    [None],
    [1U],
    VariableStringValues([]),
    fill_value=VariableStringValues(["missing"]),
  )
  writer.resize_dataset("/extensible_strings", [4UL])

  let file = Hdf5File::parse(writer.encode())
  assert_eq(string_values(file.read_dataset("/strings")), ["value", "", ""])
  assert_eq(string_values(file.read_dataset("/extensible_strings")), [
    "missing", "missing", "missing", "missing",
  ])

  let sequence = Datatype::write_vlen_sequence(Datatype::write_integer(true, 4))
  let unsupported = try {
    Hdf5Writer::new().add_extensible_dataset(
      "/sequences",
      [0UL],
      [None],
      [1U],
      StructuredValues(sequence, []),
      fill_value=StructuredValues(sequence, [Sequence([SignedInteger(1L)])]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    unsupported,
    Some(
      UnsupportedFeature(
        "custom variable-length sequence fills are not supported yet",
      ),
    ),
  )
}
