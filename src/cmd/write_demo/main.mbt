///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 2 {
    eprintln("usage: moon run cmd/write_demo -- <output.h5>")
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
    writer.add_attribute("/experiment", "revision", [], UInt64Values([1UL]))
    writer.add_attribute(
      "/experiment/counts",
      "unit",
      [],
      FixedStringValues(["items"], 8),
    )
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
