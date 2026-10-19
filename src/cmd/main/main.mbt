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
fn run(args : Array[String]) -> Int32 {
  if args.length() != 2 {
    eprintln("usage: moon run cmd/main -- <file.h5>")
    return 2
  }
  let path = args[1]
  let bytes = @fs.read_file_to_bytes(path) catch {
    _ => {
      eprintln("cannot read file: " + path)
      return 1
    }
  }
  let file = @lib.Hdf5File::parse(bytes) catch {
    error => {
      eprintln("parse error: " + error.message())
      return 1
    }
  }
  println(path + " (superblock v" + file.superblock.version.to_string() + ")")
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
        let dataset = file.open_dataset(node.path)
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
fn main {
  @sys.exit(run(@env.args()))
}
