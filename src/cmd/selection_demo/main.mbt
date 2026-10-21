///|
fn append_item(output : String, item : String, index : Int32) -> String {
  output + (if index == 0 { "" } else { ", " }) + item
}

///|
fn describe(values : @lib.DatasetValues) -> String {
  let mut output = "["
  match values {
    Integers(items) =>
      for index in 0..<items.length() {
        output = append_item(output, items[index].to_string(), index)
      }
    UnsignedIntegers(items) =>
      for index in 0..<items.length() {
        output = append_item(output, items[index].to_string(), index)
      }
    Floats(items) =>
      for index in 0..<items.length() {
        output = append_item(output, items[index].to_string(), index)
      }
    Strings(items) | Enums(items) =>
      for index in 0..<items.length() {
        output = append_item(output, items[index], index)
      }
    Structured(items) => return "structured[" + items.length().to_string() + "]"
  }
  output + "]"
}

///|
fn dimensions_text(dimensions : Array[UInt64]) -> String {
  let mut output = "["
  for index in 0..<dimensions.length() {
    output = append_item(output, dimensions[index].to_string(), index)
  }
  output + "]"
}

///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 2 {
    eprintln("usage: moon run src/cmd/selection_demo -- <file.h5>")
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
    let metadata = file.read_dataset_metadata("/matrix")
    println("matrix shape: " + dimensions_text(metadata.dataspace.dimensions))
    let slab = file.read_dataset_selection(
      "/matrix",
      @lib.Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 2UL]),
    )
    println("hyperslab: " + describe(slab.values))
    let points = file.read_dataset_selection(
      "/matrix",
      @lib.Points([[1UL, 2UL], [0UL, 1UL], [1UL, 2UL]]),
    )
    println("points: " + describe(points.values))
    0
  } catch {
    error => {
      eprintln("selection error: " + error.message())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
