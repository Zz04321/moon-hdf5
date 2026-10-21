///|
fn describe_values(values : @lib.DatasetValues) -> String {
  match values {
    Integers(items) =>
      "int[" +
      items.length().to_string() +
      "] " +
      (if items.length() > 0 { items[0].to_string() } else { "" })
    UnsignedIntegers(items) =>
      "uint[" +
      items.length().to_string() +
      "] " +
      (if items.length() > 0 { items[0].to_string() } else { "" })
    Floats(items) =>
      "float[" +
      items.length().to_string() +
      "] " +
      (if items.length() > 0 { items[0].to_string() } else { "" })
    Strings(items) =>
      "string[" +
      items.length().to_string() +
      "] " +
      (if items.length() > 0 { items[0] } else { "" })
    Enums(items) =>
      "enum[" +
      items.length().to_string() +
      "] " +
      (if items.length() > 0 { items[0] } else { "" })
    Structured(items) => "structured[" + items.length().to_string() + "]"
  }
}

///|
fn kind_name(kind : @lib.Hdf5ObjectKind) -> String {
  match kind {
    Group => "group"
    Dataset => "dataset"
    Unknown => "object"
  }
}

///|
fn dimensions_text(dimensions : Array[UInt64]) -> String {
  if dimensions.length() == 0 {
    return "scalar"
  }
  let mut output = "["
  for index in 0..<dimensions.length() {
    if index > 0 {
      output = output + ","
    }
    output = output + dimensions[index].to_string()
  }
  output + "]"
}

///|
fn maximum_dimensions_text(dimensions : Array[UInt64]) -> String {
  if dimensions.length() == 0 {
    return "scalar"
  }
  let mut output = "["
  for index in 0..<dimensions.length() {
    if index > 0 {
      output = output + ","
    }
    output = output +
      (if dimensions[index] == 0xFFFFFFFFFFFFFFFFUL {
        "unlimited"
      } else {
        dimensions[index].to_string()
      })
  }
  output + "]"
}

///|
fn datatype_name(datatype : @lib.Datatype) -> String {
  match datatype.class {
    0 => if datatype.signed { "integer" } else { "unsigned integer" }
    1 => "floating point"
    2 => "time"
    3 => "fixed string"
    4 => "bitfield"
    5 => "opaque"
    6 => "compound"
    7 => "reference"
    8 => "enum"
    9 =>
      if datatype.vlen_is_string {
        "variable string"
      } else {
        "variable sequence"
      }
    10 => "array"
    11 => "complex"
    _ => "unknown"
  }
}

///|
fn layout_name(layout : @lib.DatasetLayoutKind) -> String {
  match layout {
    Compact => "compact"
    Contiguous => "contiguous"
    Chunked => "chunked"
    Virtual => "virtual"
    Unknown => "unknown"
  }
}

///|
fn print_metadata(metadata : @lib.DatasetMetadata) -> Unit {
  println("shape: " + dimensions_text(metadata.dataspace.dimensions))
  println(
    "maximum shape: " +
    maximum_dimensions_text(metadata.dataspace.maximum_dimensions),
  )
  println(
    "datatype: " +
    datatype_name(metadata.datatype) +
    " (" +
    metadata.datatype.size.to_string() +
    " bytes)",
  )
  println("layout: " + layout_name(metadata.layout.kind()))
  let chunk_shape = metadata.layout.chunk_shape()
  if chunk_shape.length() > 0 {
    println(
      "chunks: " + dimensions_text(chunk_shape.map(value => value.to_uint64())),
    )
  }
}

///|
fn print_attributes(file : @lib.Hdf5File, path : String) -> Bool {
  let attributes = file.attributes(path) catch {
    error => {
      eprintln("attributes " + path + ": " + error.message())
      return false
    }
  }
  let mut success = true
  for attribute in attributes {
    let value = match attribute.values {
      Some(values) => describe_values(values)
      None => {
        success = false
        eprintln(
          "attribute " + path + ": " + attribute.name + ": unsupported datatype",
        )
        "<unsupported datatype>"
      }
    }
    println("  attr " + attribute.name + " = " + value)
  }
  success
}

///|
fn read_file(path : String) -> @lib.Hdf5File? {
  let bytes = @fs.read_file_to_bytes(path) catch {
    _ => {
      eprintln("cannot read file: " + path)
      return None
    }
  }
  Some(@lib.Hdf5File::parse(bytes)) catch {
    error => {
      eprintln("parse error: " + error.message())
      None
    }
  }
}

///|
fn inspect_file(path : String) -> Int32 {
  guard read_file(path) is Some(file) else { return 1 }
  println(path + " (superblock v" + file.superblock_version().to_string() + ")")
  let mut success = print_attributes(file, "/")
  let nodes = file.walk() catch {
    error => {
      eprintln("parse error: " + error.message())
      return 1
    }
  }
  for node in nodes {
    println(node.path + "  " + kind_name(node.kind))
    if node.kind == Dataset {
      try {
        let dataset = file.read_dataset(node.path)
        println("  " + describe_values(dataset.values))
      } catch {
        error => {
          eprintln("dataset " + node.path + ": " + error.message())
          success = false
        }
      }
    }
    if !print_attributes(file, node.path) {
      success = false
    }
  }
  if success {
    0
  } else {
    1
  }
}

///|
fn metadata_command(path : String, dataset_path : String) -> Int32 {
  guard read_file(path) is Some(file) else { return 1 }
  try {
    print_metadata(file.read_dataset_metadata(dataset_path))
    0
  } catch {
    error => {
      eprintln("dataset " + dataset_path + ": " + error.message())
      1
    }
  }
}

///|
fn parse_uint(raw : String) -> UInt64? {
  if raw.length() == 0 {
    return None
  }
  let mut value = 0UL
  for index in 0..<raw.length() {
    let byte = raw[index].to_int()
    if byte < 48 || byte > 57 {
      return None
    }
    let digit = (byte - 48).to_uint64()
    if value > (0xFFFFFFFFFFFFFFFFUL - digit) / 10UL {
      return None
    }
    value = value * 10UL + digit
  }
  Some(value)
}

///|
fn parse_coordinate(raw : String) -> Array[UInt64]? {
  if raw == "" {
    return Some([])
  }
  let coordinate : Array[UInt64] = []
  for item in raw.split(",") {
    guard parse_uint(item.to_owned()) is Some(value) else { return None }
    coordinate.push(value)
  }
  Some(coordinate)
}

///|
fn parse_points(raw : String) -> Array[Array[UInt64]]? {
  if raw == "" {
    return Some([])
  }
  let points : Array[Array[UInt64]] = []
  for item in raw.split(";") {
    guard parse_coordinate(item.to_owned()) is Some(point) else { return None }
    points.push(point)
  }
  Some(points)
}

///|
fn append_preview(output : String, value : String, index : Int32) -> String {
  output + (if index == 0 { "" } else { ", " }) + value
}

///|
fn preview_values(values : @lib.DatasetValues) -> String {
  let mut output = "["
  match values {
    Integers(items) =>
      for index in 0..<items.length().min(16) {
        output = append_preview(output, items[index].to_string(), index)
      }
    UnsignedIntegers(items) =>
      for index in 0..<items.length().min(16) {
        output = append_preview(output, items[index].to_string(), index)
      }
    Floats(items) =>
      for index in 0..<items.length().min(16) {
        output = append_preview(output, items[index].to_string(), index)
      }
    Strings(items) | Enums(items) =>
      for index in 0..<items.length().min(16) {
        output = append_preview(output, "\"" + items[index] + "\"", index)
      }
    Structured(items) => return "structured[" + items.length().to_string() + "]"
  }
  output + "]"
}

///|
fn print_selection(result : @lib.DatasetSelectionResult) -> Unit {
  println("shape: " + dimensions_text(result.dimensions))
  println("values: " + preview_values(result.values))
}

///|
fn read_command(args : Array[String]) -> Int32 {
  if args.length() != 7 && args.length() != 6 {
    return 2
  }
  guard read_file(args[2]) is Some(file) else { return 1 }
  let selection = if args.length() == 6 && args[4] == "--points" {
    guard parse_points(args[5]) is Some(points) else {
      eprintln("invalid point coordinates")
      return 2
    }
    @lib.Points(points)
  } else if args.length() == 7 {
    guard parse_coordinate(args[4]) is Some(start) &&
      parse_coordinate(args[5]) is Some(count) &&
      parse_coordinate(args[6]) is Some(stride) else {
      eprintln("invalid hyperslab coordinates")
      return 2
    }
    @lib.Hyperslab(start, count, stride)
  } else {
    return 2
  }
  try {
    print_selection(file.read_dataset_selection(args[3], selection))
    0
  } catch {
    error => {
      eprintln("dataset " + args[3] + ": " + error.message())
      1
    }
  }
}

///|
fn float_stats(values : Array[Double]) -> String {
  if values.length() == 0 {
    return "count=0"
  }
  let mut minimum = values[0]
  let mut maximum = values[0]
  let mut sum = 0.0
  for value in values {
    if value < minimum {
      minimum = value
    }
    if value > maximum {
      maximum = value
    }
    sum = sum + value
  }
  "count=" +
  values.length().to_string() +
  " min=" +
  minimum.to_string() +
  " max=" +
  maximum.to_string() +
  " mean=" +
  (sum / values.length().to_double()).to_string()
}

///|
fn integer_stats(values : Array[Int64]) -> String {
  if values.length() == 0 {
    return "count=0"
  }
  let mut minimum = values[0]
  let mut maximum = values[0]
  let mut sum = 0.0
  for value in values {
    if value < minimum {
      minimum = value
    }
    if value > maximum {
      maximum = value
    }
    sum = sum + value.to_double()
  }
  "count=" +
  values.length().to_string() +
  " min=" +
  minimum.to_string() +
  " max=" +
  maximum.to_string() +
  " mean=" +
  (sum / values.length().to_double()).to_string()
}

///|
fn unsigned_stats(values : Array[UInt64]) -> String {
  if values.length() == 0 {
    return "count=0"
  }
  let mut minimum = values[0]
  let mut maximum = values[0]
  let mut sum = 0.0
  for value in values {
    if value < minimum {
      minimum = value
    }
    if value > maximum {
      maximum = value
    }
    sum = sum + value.to_double()
  }
  "count=" +
  values.length().to_string() +
  " min=" +
  minimum.to_string() +
  " max=" +
  maximum.to_string() +
  " mean=" +
  (sum / values.length().to_double()).to_string()
}

///|
fn stats_text(values : @lib.DatasetValues) -> String {
  match values {
    Integers(items) => integer_stats(items)
    UnsignedIntegers(items) => unsigned_stats(items)
    Floats(items) => float_stats(items)
    Strings(items) => {
      let mut empty = 0
      for item in items {
        if item == "" {
          empty = empty + 1
        }
      }
      "count=" + items.length().to_string() + " empty=" + empty.to_string()
    }
    Enums(items) => "count=" + items.length().to_string()
    Structured(items) => "count=" + items.length().to_string()
  }
}

///|
fn stats_command(path : String, dataset_path : String) -> Int32 {
  guard read_file(path) is Some(file) else { return 1 }
  try {
    let dataset = file.read_dataset(dataset_path)
    println(stats_text(dataset.values))
    0
  } catch {
    error => {
      eprintln("dataset " + dataset_path + ": " + error.message())
      1
    }
  }
}

///|
fn print_usage() -> Unit {
  eprintln("usage:")
  eprintln("  moon run src/cmd/main -- <file.h5>")
  eprintln("  moon run src/cmd/main -- inspect <file.h5>")
  eprintln("  moon run src/cmd/main -- metadata <file.h5> <dataset>")
  eprintln("  moon run src/cmd/main -- stats <file.h5> <dataset>")
  eprintln(
    "  moon run src/cmd/main -- read <file.h5> <dataset> <start> <count> <stride>",
  )
  eprintln(
    "  moon run src/cmd/main -- read <file.h5> <dataset> --points <p1;p2;...>",
  )
}

///|
fn run(args : Array[String]) -> Int32 {
  let status = if args.length() == 2 {
    inspect_file(args[1])
  } else if args.length() == 3 && args[1] == "inspect" {
    inspect_file(args[2])
  } else if args.length() == 4 && args[1] == "metadata" {
    metadata_command(args[2], args[3])
  } else if args.length() == 4 && args[1] == "stats" {
    stats_command(args[2], args[3])
  } else if args.length() >= 2 && args[1] == "read" {
    read_command(args)
  } else {
    2
  }
  if status == 2 {
    print_usage()
  }
  status
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
