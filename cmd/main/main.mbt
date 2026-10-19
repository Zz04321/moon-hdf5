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
fn print_attributes(file : @lib.Hdf5File, path : String) -> Unit {
  let attributes = file.attributes(path) catch { _ => [] }
  for attribute in attributes {
    let value = match attribute.values {
      Some(values) => describe_values(values)
      None => "<unsupported datatype>"
    }
    println("  attr " + attribute.name + " = " + value)
  }
}

///|
fn main {
  let args = @env.args()
  if args.length() < 2 {
    println("usage: moon run cmd/main -- <file.h5>")
    return
  }
  let path = args[1]
  let bytes = @fs.read_file_to_bytes(path) catch {
    _ => {
      println("cannot read file: " + path)
      return
    }
  }
  let file = @lib.Hdf5File::parse(bytes) catch {
    error => {
      println("parse error: " + error.message())
      return
    }
  }
  println(path + " (superblock v" + file.superblock.version.to_string() + ")")
  print_attributes(file, "/")
  let nodes = file.walk() catch {
    error => {
      println("parse error: " + error.message())
      return
    }
  }
  for node in nodes {
    println(node.path + "  " + kind_name(node.kind))
    if node.kind == Dataset {
      try {
        let dataset = file.open_dataset(node.path)
        println("  " + describe_values(dataset.values))
      } catch {
        error => println("  " + error.message())
      }
    }
    print_attributes(file, node.path)
  }
}
