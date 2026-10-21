///|
fn dimensions_text(dimensions : Array[UInt64]) -> String {
  if dimensions.length() == 0 {
    return "scalar"
  }
  let mut output = "["
  for index in 0..<dimensions.length() {
    output = output +
      (if index == 0 { "" } else { ", " }) +
      dimensions[index].to_string()
  }
  output + "]"
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
fn run(args : Array[String]) -> Int32 {
  if args.length() != 3 {
    eprintln(
      "usage: moon run src/cmd/metadata_demo -- <file.h5> <dataset-path>",
    )
    return 2
  }
  let bytes = @fs.read_file_to_bytes(args[1]) catch {
    _ => {
      eprintln("cannot read file: " + args[1])
      return 1
    }
  }
  try {
    let file = @lib.Hdf5File::parse(bytes)
    let metadata = file.read_dataset_metadata(args[2])
    println("dataset: " + args[2])
    println("shape: " + dimensions_text(metadata.dataspace.dimensions))
    println("datatype class: " + metadata.datatype.class.to_string())
    println("element bytes: " + metadata.datatype.size.to_string())
    println("layout: " + layout_name(metadata.layout.kind()))
    println("attributes: " + file.attributes(args[2]).length().to_string())
    0
  } catch {
    error => {
      eprintln("metadata error: " + error.message())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
