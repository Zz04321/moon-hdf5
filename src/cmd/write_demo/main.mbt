///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 2 {
    eprintln("usage: moon run src/cmd/write_demo -- <output.h5>")
    return 2
  }
  let writer = @lib.Hdf5Writer::new()
  try {
    writer.add_group("/experiment")
    writer.add_dataset(
      "/experiment/counts",
      [2UL, 3UL],
      Int64Values([1L, 2L, 3L, 4L, 5L, 6L]),
    )
    writer.add_dataset(
      "/experiment/reference_target",
      [2UL],
      Int64Values([11L, 12L]),
    )
    writer.add_dataset(
      "/experiment/region_target",
      [6UL],
      Int64Values([10L, 11L, 12L, 13L, 14L, 15L]),
    )
    let region_ref = @lib.Datatype::write_region_reference()
    writer.add_dataset(
      "/experiment/region_references",
      [7UL],
      StructuredValues(region_ref, [
        @lib.RegionReference(
          Some("/experiment/region_target"),
          @lib.RegionPoints([[4UL], [0UL], [3UL]]),
        ),
        @lib.RegionReference(
          Some("/experiment/region_target"),
          @lib.RegionHyperslab([1UL], [1UL], [1UL], [2UL]),
        ),
        @lib.RegionReference(
          Some("/experiment/region_target"),
          @lib.RegionHyperslab([0UL], [3UL], [2UL], [2UL]),
        ),
        @lib.RegionReference(
          Some("/experiment/region_target"),
          @lib.RegionBlocks([
            { start: [0UL], end: [1UL], },
            { start: [4UL], end: [4UL], },
          ]),
        ),
        @lib.RegionReference(Some("/experiment/region_target"), @lib.RegionAll),
        @lib.RegionReference(
          Some("/experiment/region_target"),
          @lib.RegionEmpty,
        ),
        @lib.RegionReference(None, @lib.RegionEmpty),
      ]),
    )
    writer.add_attribute(
      "/experiment/region_target",
      "window",
      [],
      StructuredValues(region_ref, [
        @lib.RegionReference(
          Some("/experiment/region_target"),
          @lib.RegionHyperslab([2UL], [1UL], [1UL], [3UL]),
        ),
      ]),
    )
    writer.add_dataset(
      "/experiment/references",
      [4UL],
      ObjectReferenceValues([
        Some("/experiment/reference_target"),
        Some("/experiment"),
        None,
        Some("/"),
      ]),
    )
    writer.add_attribute(
      "/experiment/reference_target",
      "self_ref",
      [],
      ObjectReferenceValues([Some("/experiment/reference_target")]),
    )
    writer.add_soft_link("/experiment", "latest", "/experiment/counts")
    writer.add_external_link(
      "/experiment", "external", "external-target.h5", "/experiment/counts",
    )
    writer.add_external_dataset(
      "/experiment/external_raw",
      [6UL],
      Int32Values([10, 20, 30, 40, 50, 60]),
      [
        {
          file_name: "moon-hdf5-demo-external-0.bin",
          file_offset: 4UL,
          data_size: 8UL,
        },
        {
          file_name: "moon-hdf5-demo-external-1.bin",
          file_offset: 0UL,
          data_size: 16UL,
        },
      ],
    )
    writer.add_dataset(
      "/experiment/labels",
      [2UL],
      FixedStringValues(["control", "sample"], 8),
    )
    writer.add_dataset(
      "/experiment/metrics",
      [2UL],
      Float64Values([1.5, -2.25]),
    )
    let sparse = Array::make(35, -7L)
    sparse[0] = 42L
    sparse[17] = 99L
    sparse[34] = -2L
    writer.add_chunked_dataset(
      "/experiment/chunked",
      [5UL, 7UL],
      [2U, 3U],
      Int64Values(sparse),
      fill_value=Int64Values([-7L]),
      filters=[Shuffle, Deflate(6), Fletcher32],
    )
    writer.add_extensible_dataset(
      "/experiment/series",
      [0UL, 2UL],
      [None, Some(2UL)],
      [2U, 2U],
      Int64Values([]),
      fill_value=Int64Values([-1L]),
      filters=[Shuffle, Deflate(6), Fletcher32],
    )
    writer.append_dataset(
      "/experiment/series",
      [2UL, 2UL],
      Int64Values([10L, 11L, 20L, 21L]),
    )
    writer.append_dataset(
      "/experiment/series",
      [1UL, 2UL],
      Int64Values([30L, 31L]),
    )
    writer.resize_dataset("/experiment/series", [4UL, 2UL])
    let paged = Array::make(134200, 0U)
    paged[0] = 1U
    paged[243] = 2U
    paged[244] = 3U
    paged[131059] = 4U
    paged[131060] = 5U
    paged[132100] = 6U
    paged[134199] = 7U
    writer.add_extensible_dataset(
      "/experiment/paged_series",
      [134200UL],
      [None],
      [1U],
      UInt8Values(paged),
      filters=[Deflate(6), Fletcher32],
    )
    writer.add_dataset(
      "/experiment/names",
      [4UL],
      VariableStringValues(["月亮", "HDF5", "", "月亮"]),
    )
    writer.add_chunked_dataset(
      "/experiment/chunked_names",
      [5UL],
      [2U],
      VariableStringValues(["alpha", "", "数据", "alpha", ""]),
    )
    writer.add_extensible_dataset(
      "/experiment/bounded_names",
      [1UL, 2UL],
      [None, Some(4UL)],
      [1U, 2U],
      VariableStringValues(["a", "b"]),
    )
    writer.add_extensible_dataset(
      "/experiment/events",
      [0UL],
      [None],
      [2U],
      VariableStringValues([]),
    )
    writer.append_dataset(
      "/experiment/events",
      [2UL],
      VariableStringValues(["start", "结束"]),
    )
    writer.resize_dataset("/experiment/events", [3UL])
    writer.add_chunked_dataset(
      "/experiment/string_fill",
      [3UL],
      [2U],
      VariableStringValues(["value", "", ""]),
      fill_value=VariableStringValues(["missing"]),
    )
    writer.add_extensible_dataset(
      "/experiment/fill_names",
      [0UL],
      [None],
      [1U],
      VariableStringValues([]),
      fill_value=VariableStringValues(["missing"]),
    )
    writer.resize_dataset("/experiment/fill_names", [4UL])
    writer.add_attribute("/experiment", "revision", [], UInt64Values([1UL]))
    writer.add_attribute(
      "/experiment/counts",
      "unit",
      [],
      FixedStringValues(["items"], 8),
    )
    writer.add_attribute(
      "/experiment/names",
      "description",
      [],
      VariableStringValues(["UTF-8 标签"]),
    )
    writer.add_group("/types")
    writer.add_attribute("/types", "i8_marker", [], Int8Values([-5]))
    writer.add_attribute(
      "/types",
      "f32_marker",
      [],
      Float32Values([Float(1.5)]),
    )
    writer.add_attribute(
      "/types",
      "big_i32_marker",
      [],
      StructuredValues(
        @lib.Datatype::write_integer_with_order(true, 4, false),
        [@lib.SignedInteger(-5L)],
      ),
    )
    writer.add_attribute(
      "/types",
      "complex_marker",
      [],
      Complex64Values([(-1.5, 2.5)]),
    )
    writer.add_dataset("/types/i8", [3UL], Int8Values([-128, 0, 127]))
    writer.add_dataset("/types/i16", [3UL], Int16Values([-32768, 0, 32767]))
    writer.add_dataset(
      "/types/i32",
      [3UL],
      Int32Values([-2147483647, 0, 2147483647]),
    )
    writer.add_dataset("/types/u8", [3UL], UInt8Values([0U, 1U, 255U]))
    writer.add_dataset("/types/u16", [3UL], UInt16Values([0U, 1U, 65535U]))
    writer.add_dataset("/types/u32", [3UL], UInt32Values([0U, 1U, 0xFFFFFFFFU]))
    writer.add_dataset(
      "/types/f32",
      [3UL],
      Float32Values([Float(-1.5), Float(0.0), Float(2.25)]),
    )
    writer.add_dataset(
      "/types/complex64",
      [2UL],
      Complex64Values([(1.5, -2.25), (-3.0, 4.5)]),
    )
    writer.add_dataset(
      "/types/complex128",
      [2UL],
      Complex128Values([(5.5, -6.25), (-7.0, 8.75)]),
    )
    writer.add_chunked_dataset(
      "/types/chunked_complex64",
      [4UL],
      [2U],
      Complex64Values([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]),
      fill_value=Complex64Values([(-1.0, -2.0)]),
      filters=[Shuffle, Deflate(6)],
    )
    writer.add_extensible_dataset(
      "/types/growing_complex128",
      [0UL],
      [None],
      [2U],
      Complex128Values([]),
      fill_value=Complex128Values([(-1.0, -2.0)]),
    )
    writer.append_dataset(
      "/types/growing_complex128",
      [2UL],
      Complex128Values([(9.0, 10.0), (11.0, 12.0)]),
    )
    writer.resize_dataset("/types/growing_complex128", [3UL])
    let big_i16 = @lib.Datatype::write_integer_with_order(true, 2, false)
    writer.add_dataset(
      "/types/big_i16",
      [3UL],
      StructuredValues(big_i16, [
        @lib.SignedInteger(-32768L),
        @lib.SignedInteger(-1L),
        @lib.SignedInteger(12345L),
      ]),
    )
    writer.add_dataset(
      "/types/big_u64",
      [3UL],
      StructuredValues(
        @lib.Datatype::write_integer_with_order(false, 8, false),
        [
          @lib.UnsignedInteger(0UL),
          @lib.UnsignedInteger(1UL << 40),
          @lib.UnsignedInteger(0xFFFFFFFFFFFFFFFFUL),
        ],
      ),
    )
    writer.add_dataset(
      "/types/big_f32",
      [3UL],
      StructuredValues(@lib.Datatype::write_float_with_order(4, false), [
        @lib.FloatingPoint(-1.5),
        @lib.FloatingPoint(0.0),
        @lib.FloatingPoint(2.25),
      ]),
    )
    writer.add_dataset(
      "/types/big_f64",
      [3UL],
      StructuredValues(@lib.Datatype::write_float_with_order(8, false), [
        @lib.FloatingPoint(-1.25),
        @lib.FloatingPoint(0.0),
        @lib.FloatingPoint(3.5),
      ]),
    )
    let big_state = @lib.Datatype::write_enum(
      @lib.Datatype::write_integer_with_order(true, 2, false),
      [
        { name: "LOW", value: @lib.EnumValue::Signed(-1L), },
        { name: "HIGH", value: @lib.EnumValue::Signed(300L), },
      ],
    )
    writer.add_dataset(
      "/types/big_state",
      [2UL],
      StructuredValues(big_state, [
        @lib.Enumeration("HIGH"),
        @lib.Enumeration("LOW"),
      ]),
    )
    let big_samples = @lib.Datatype::write_array(
      @lib.Datatype::write_integer_with_order(false, 4, false),
      [2UL],
    )
    let big_record_type = @lib.Datatype::write_compound([
      ("id", @lib.Datatype::write_integer(true, 4)),
      ("weight", @lib.Datatype::write_float_with_order(8, false)),
      ("count", @lib.Datatype::write_integer_with_order(false, 2, false)),
      ("state", big_state),
      ("samples", big_samples),
    ])
    writer.add_dataset(
      "/types/big_record",
      [1UL],
      StructuredValues(big_record_type, [
        @lib.Compound([
          { name: "id", value: @lib.SignedInteger(7L), },
          { name: "weight", value: @lib.FloatingPoint(-2.5), },
          { name: "count", value: @lib.UnsignedInteger(513UL), },
          { name: "state", value: @lib.Enumeration("HIGH"), },
          {
            name: "samples",
            value: @lib.Elements([
              @lib.UnsignedInteger(1UL),
              @lib.UnsignedInteger(0x10203040UL),
            ]),
          },
        ]),
      ]),
    )
    writer.add_extensible_dataset(
      "/types/big_fill",
      [1UL],
      [None],
      [2U],
      StructuredValues(big_i16, [@lib.SignedInteger(7L)]),
      fill_value=StructuredValues(big_i16, [@lib.SignedInteger(-999L)]),
    )
    writer.resize_dataset("/types/big_fill", [4UL])
    let unix_time = @lib.Datatype::write_time(8)
    writer.add_dataset(
      "/types/time",
      [3UL],
      StructuredValues(unix_time, [
        @lib.Time(0UL),
        @lib.Time(1700000000UL),
        @lib.Time(42UL),
      ]),
    )
    writer.add_extensible_dataset(
      "/types/time_fill",
      [1UL],
      [None],
      [2U],
      StructuredValues(unix_time, [@lib.Time(7UL)]),
      fill_value=StructuredValues(unix_time, [@lib.Time(99UL)]),
    )
    writer.resize_dataset("/types/time_fill", [3UL])
    writer.add_dataset(
      "/types/bitfield",
      [3UL],
      StructuredValues(@lib.Datatype::write_bitfield(2, 12), [
        @lib.Bitfield(0UL),
        @lib.Bitfield(240UL),
        @lib.Bitfield(0xFFFUL),
      ]),
    )
    writer.add_chunked_dataset(
      "/types/opaque",
      [2UL],
      [1U],
      StructuredValues(@lib.Datatype::write_opaque(4, "MOON|V4"), [
        @lib.Opaque(b"\x00\x01\xfe\xff"),
        @lib.Opaque(b"MOON"),
      ]),
      filters=[Shuffle, Deflate(6)],
    )
    writer.add_chunked_dataset(
      "/types/masked_chunks",
      [8UL],
      [2U],
      Int64Values([11L, 12L, 21L, 22L, 31L, 32L, 41L, 42L]),
      filters=[Shuffle, Deflate(6)],
    )
    let chunk_masks : Array[UInt32] = [0U, 1U, 2U, 3U]
    for index in 0..<chunk_masks.length() {
      writer.set_chunk_filter_mask(
        "/types/masked_chunks",
        index,
        chunk_masks[index],
      )
    }
    let int_sequence = @lib.Datatype::write_vlen_sequence(
      @lib.Datatype::write_integer(true, 4),
    )
    writer.add_dataset(
      "/types/sequences",
      [4UL],
      StructuredValues(int_sequence, [
        @lib.Sequence([]),
        @lib.Sequence([
          @lib.SignedInteger(1L),
          @lib.SignedInteger(-2L),
          @lib.SignedInteger(3L),
        ]),
        @lib.Sequence([
          @lib.SignedInteger(1L),
          @lib.SignedInteger(-2L),
          @lib.SignedInteger(3L),
        ]),
        @lib.Sequence([@lib.SignedInteger(-1L), @lib.SignedInteger(0L)]),
      ]),
    )
    writer.add_attribute(
      "/types/sequences",
      "prototype",
      [],
      StructuredValues(int_sequence, [
        @lib.Sequence([@lib.SignedInteger(7L), @lib.SignedInteger(8L)]),
      ]),
    )
    writer.add_chunked_dataset(
      "/types/chunked_i16",
      [5UL],
      [2U],
      Int16Values([-7, 2, -7, 4, -7]),
      fill_value=Int16Values([-7]),
      filters=[Shuffle, Deflate(6), Fletcher32],
    )
    writer.add_chunked_dataset(
      "/types/scaleoffset_i32",
      [8UL],
      [4U],
      Int32Values([10, -7, 12, 13, 100, -7, 101, 102]),
      fill_value=Int32Values([-7]),
      filters=[ScaleOffset(0)],
    )
    writer.add_chunked_dataset(
      "/types/scaleoffset_pipeline",
      [6UL],
      [3U],
      Int16Values([100, 101, 102, 200, 201, 202]),
      fill_value=Int16Values([-1]),
      filters=[ScaleOffset(4), Deflate(6), Fletcher32],
    )
    writer.add_chunked_dataset(
      "/types/scaleoffset_u64",
      [3UL],
      [3U],
      UInt64Values([1UL << 40, (1UL << 40) + 1UL, (1UL << 40) + 2UL]),
      fill_value=UInt64Values([7UL]),
      filters=[ScaleOffset(0)],
    )
    writer.add_chunked_dataset(
      "/types/scaleoffset_full",
      [4UL],
      [2U],
      Int32Values([-5, -4, -3, -2]),
      fill_value=Int32Values([0]),
      filters=[ScaleOffset(32)],
    )
    writer.add_chunked_dataset(
      "/types/scaleoffset_f32",
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
      "/types/scaleoffset_f64",
      [5UL],
      [5U],
      Float64Values([1.5, -2.5, 3.5, -9.5, 0.0]),
      fill_value=Float64Values([-9.5]),
      filters=[ScaleOffset(1)],
    )
    writer.add_chunked_dataset(
      "/types/scaleoffset_f32_pipeline",
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
    writer.add_extensible_dataset(
      "/types/growing_f32",
      [0UL],
      [None],
      [2U],
      Float32Values([]),
      fill_value=Float32Values([Float(-1.0)]),
      filters=[Shuffle, Deflate(6), Fletcher32],
    )
    writer.append_dataset(
      "/types/growing_f32",
      [3UL],
      Float32Values([Float(1.25), Float(2.5), Float(3.75)]),
    )
    writer.resize_dataset("/types/growing_f32", [4UL])
    writer.add_group("/structured")
    let state_type = @lib.Datatype::write_enum(
      @lib.Datatype::write_integer(false, 1),
      [
        { name: "UNKNOWN", value: @lib.EnumValue::Unsigned(0UL), },
        { name: "OPEN", value: @lib.EnumValue::Unsigned(1UL), },
        { name: "CLOSED", value: @lib.EnumValue::Unsigned(2UL), },
      ],
    )
    let samples_type = @lib.Datatype::write_array(
      @lib.Datatype::write_integer(true, 2),
      [2UL, 2UL],
    )
    let record_type = @lib.Datatype::write_compound([
      ("id", @lib.Datatype::write_integer(true, 4)),
      ("state", state_type),
      ("samples", samples_type),
      ("label", @lib.Datatype::write_fixed_string(6)),
    ])
    let empty_record = @lib.Compound([
      { name: "id", value: @lib.SignedInteger(0L), },
      { name: "state", value: @lib.Enumeration("UNKNOWN"), },
      {
        name: "samples",
        value: @lib.Elements([
          @lib.SignedInteger(0L),
          @lib.SignedInteger(0L),
          @lib.SignedInteger(0L),
          @lib.SignedInteger(0L),
        ]),
      },
      { name: "label", value: @lib.Text(""), },
    ])
    let first_record = @lib.Compound([
      { name: "id", value: @lib.SignedInteger(7L), },
      { name: "state", value: @lib.Enumeration("OPEN"), },
      {
        name: "samples",
        value: @lib.Elements([
          @lib.SignedInteger(1L),
          @lib.SignedInteger(2L),
          @lib.SignedInteger(3L),
          @lib.SignedInteger(4L),
        ]),
      },
      { name: "label", value: @lib.Text("alpha"), },
    ])
    let second_record = @lib.Compound([
      { name: "id", value: @lib.SignedInteger(-2L), },
      { name: "state", value: @lib.Enumeration("CLOSED"), },
      {
        name: "samples",
        value: @lib.Elements([
          @lib.SignedInteger(-1L),
          @lib.SignedInteger(-2L),
          @lib.SignedInteger(-3L),
          @lib.SignedInteger(-4L),
        ]),
      },
      { name: "label", value: @lib.Text("beta"), },
    ])
    writer.add_chunked_dataset(
      "/structured/records",
      [3UL],
      [2U],
      StructuredValues(record_type, [first_record, empty_record, second_record]),
      fill_value=StructuredValues(record_type, [empty_record]),
      filters=[Shuffle, Deflate(6), Fletcher32],
    )
    writer.add_attribute(
      "/structured/records",
      "prototype",
      [],
      StructuredValues(record_type, [first_record]),
    )
    let object_ref = @lib.Datatype::write_object_reference()
    let reference_pair = @lib.Datatype::write_array(object_ref, [2UL])
    let reference_history = @lib.Datatype::write_vlen_sequence(object_ref)
    let reference_record_type = @lib.Datatype::write_compound([
      ("label", @lib.Datatype::write_fixed_string(6)),
      ("target", object_ref),
      ("pair", reference_pair),
    ])
    writer.add_dataset(
      "/structured/reference_records",
      [2UL],
      StructuredValues(reference_record_type, [
        @lib.Compound([
          { name: "label", value: @lib.Text("first"), },
          {
            name: "target",
            value: @lib.ObjectReference(Some("/experiment/reference_target")),
          },
          {
            name: "pair",
            value: @lib.Elements([
              @lib.ObjectReference(Some("/experiment")),
              @lib.ObjectReference(None),
            ]),
          },
        ]),
        @lib.Compound([
          { name: "label", value: @lib.Text("second"), },
          { name: "target", value: @lib.ObjectReference(Some("/")), },
          {
            name: "pair",
            value: @lib.Elements([
              @lib.ObjectReference(Some("/experiment/reference_target")),
              @lib.ObjectReference(Some("/structured")),
            ]),
          },
        ]),
      ]),
    )
    writer.add_dataset(
      "/structured/reference_sequences",
      [2UL],
      StructuredValues(reference_history, [
        @lib.Sequence([
          @lib.ObjectReference(Some("/experiment/reference_target")),
          @lib.ObjectReference(Some("/experiment")),
        ]),
        @lib.Sequence([@lib.ObjectReference(None)]),
      ]),
    )
    let region_pair = @lib.Datatype::write_array(region_ref, [2UL])
    let region_record_type = @lib.Datatype::write_compound([
      ("label", @lib.Datatype::write_fixed_string(6)),
      ("target", region_ref),
      ("pair", region_pair),
    ])
    let region_record = @lib.Compound([
      { name: "label", value: @lib.Text("window"), },
      {
        name: "target",
        value: @lib.RegionReference(
          Some("/experiment/region_target"),
          @lib.RegionPoints([[2UL]]),
        ),
      },
      {
        name: "pair",
        value: @lib.Elements([
          @lib.RegionReference(
            Some("/experiment/region_target"),
            @lib.RegionHyperslab([1UL], [1UL], [1UL], [2UL]),
          ),
          @lib.RegionReference(None, @lib.RegionEmpty),
        ]),
      },
    ])
    writer.add_dataset(
      "/structured/region_records",
      [1UL],
      StructuredValues(region_record_type, [region_record]),
    )
    writer.add_attribute(
      "/structured/region_records",
      "prototype",
      [],
      StructuredValues(region_record_type, [region_record]),
    )
    let vector_type = @lib.Datatype::write_array(
      @lib.Datatype::write_float(4),
      [2UL],
    )
    writer.add_dataset(
      "/structured/vectors",
      [2UL],
      StructuredValues(vector_type, [
        @lib.Elements([@lib.FloatingPoint(1.5), @lib.FloatingPoint(-2.0)]),
        @lib.Elements([@lib.FloatingPoint(3.25), @lib.FloatingPoint(4.5)]),
      ]),
    )
    writer.add_extensible_dataset(
      "/structured/states",
      [0UL],
      [None],
      [2U],
      StructuredValues(state_type, []),
      fill_value=StructuredValues(state_type, [@lib.Enumeration("UNKNOWN")]),
    )
    writer.append_dataset(
      "/structured/states",
      [2UL],
      StructuredValues(state_type, [
        @lib.Enumeration("OPEN"),
        @lib.Enumeration("CLOSED"),
      ]),
    )
    writer.resize_dataset("/structured/states", [4UL])
    writer.add_group("/metadata")
    writer.add_dataset("/metadata/values", [], UInt64Values([42UL]))
    for index in 0..<2500 {
      writer.add_attribute(
        "/metadata/values",
        "attr_" + index.to_string(),
        [],
        UInt64Values([index.to_uint64()]),
      )
    }
    writer.add_group("/metadata/dense")
    for index in 0..<12 {
      writer.add_dataset(
        "/metadata/dense/value_" + index.to_string(),
        [],
        UInt64Values([index.to_uint64()]),
      )
      writer.add_attribute(
        "/metadata/dense",
        "dense_attr_" + index.to_string(),
        [],
        UInt64Values([(index + 100).to_uint64()]),
      )
    }
    writer.add_attribute(
      "/metadata/dense",
      "dense_text",
      [],
      VariableStringValues(["密集属性"]),
    )
    writer.add_group("/metadata/ordered")
    let ordered_links = [
      "zeta", "alpha", "mu", "beta", "omega", "delta", "theta", "gamma", "sigma",
      "kappa",
    ]
    for index in 0..<ordered_links.length() {
      writer.add_dataset(
        "/metadata/ordered/" + ordered_links[index],
        [],
        UInt64Values([index.to_uint64()]),
      )
    }
    let ordered_attributes = [
      "version", "units", "title", "owner", "status", "source", "time", "kind", "tag",
      "note",
    ]
    for index in 0..<ordered_attributes.length() {
      writer.add_attribute(
        "/metadata/ordered",
        ordered_attributes[index],
        [],
        UInt64Values([index.to_uint64()]),
      )
    }
    writer.add_dataset("/metadata/continued", [], UInt64Values([0UL]))
    let continued_value = "c".repeat(9000)
    for index in 0..<8 {
      writer.add_attribute(
        "/metadata/continued",
        "continued_" + index.to_string(),
        [],
        FixedStringValues([continued_value], 9000),
      )
    }
    writer.add_group("/metadata/continued_links")
    let continued_target = "/" + "t".repeat(9000)
    for index in 0..<8 {
      writer.add_soft_link(
        "/metadata/continued_links",
        "link_" + index.to_string(),
        continued_target,
      )
    }
    writer.add_dataset("/metadata/continued_multi", [], UInt64Values([1UL]))
    let multi_value = "m".repeat(60000)
    for index in 0..<20 {
      writer.add_attribute(
        "/metadata/continued_multi",
        "multi_" + index.to_string(),
        [],
        FixedStringValues([multi_value], 60000),
      )
    }
    @fs.write_bytes_to_file(args[1], writer.encode())
    println("wrote " + args[1])
    0
  } catch {
    error => {
      eprintln("write error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
