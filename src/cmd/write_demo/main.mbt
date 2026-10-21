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
      "/experiment/filtered_names",
      [5UL],
      [2U],
      VariableStringValues(["alpha", "", "数据", "alpha", ""]),
      filters=[Deflate(6), Fletcher32],
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
    writer.add_chunked_dataset(
      "/types/chunked_i16",
      [5UL],
      [2U],
      Int16Values([-7, 2, -7, 4, -7]),
      fill_value=Int16Values([-7]),
      filters=[Shuffle, Deflate(6), Fletcher32],
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
