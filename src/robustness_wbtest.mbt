///|
struct MutationRng {
  mut state : UInt64
}

///|
fn MutationRng::next(self : MutationRng) -> UInt64 {
  let mut value = self.state
  value = value ^ (value << 13)
  value = value ^ (value >> 7)
  value = value ^ (value << 17)
  self.state = value
  value
}

///|
fn MutationRng::below(self : MutationRng, bound : Int32) -> Int32 {
  if bound <= 1 {
    0
  } else {
    (self.next() % bound.to_uint64()).to_int()
  }
}

///|
fn mutate_fixture(source : Bytes, rng : MutationRng) -> Bytes {
  let bytes = source.to_fixedarray()
  let length = bytes.length()
  if length == 0 {
    return Bytes::from_array(bytes)
  }
  let changes = 1 + rng.below(8)
  for _ in 0..<changes {
    let position = rng.below(length)
    match rng.below(5) {
      0 =>
        bytes[position] = (bytes[position].to_int() ^ (1 << rng.below(8))).to_byte()
      1 => bytes[position] = rng.below(256).to_byte()
      2 => {
        let mut run = 1 + rng.below(16)
        if position + run > length {
          run = length - position
        }
        for index in position..<(position + run) {
          bytes[index] = b'\x00'
        }
      }
      3 => {
        let mut run = 1 + rng.below(16)
        if position + run > length {
          run = length - position
        }
        for index in position..<(position + run) {
          bytes[index] = b'\xFF'
        }
      }
      _ => bytes[position] = bytes[rng.below(length)]
    }
  }
  Bytes::from_array(bytes)
}

///|
fn truncate_fixture(source : Bytes, length : Int32) -> Bytes {
  source.get_view(start=0, end=length).unwrap().to_owned()
}

///|
fn exercise_bytes(source : Bytes) -> Unit {
  let file = Hdf5File::parse(source) catch { _ => return }
  let nodes = file.walk() catch { _ => return }
  for node in nodes {
    try file.attributes(node.path) |> ignore catch {
      _ => ()
    }
    if node.kind == Dataset {
      try file.read_dataset(node.path) |> ignore catch {
        _ => ()
      }
    }
  }
}

///|
fn robustness_fixtures() -> Array[String] {
  [
    "testdata/basic_v0.h5", "testdata/latest_v3.h5", "testdata/attribute_storage.h5",
    "testdata/group_storage.h5", "testdata/fixed_array.h5", "testdata/extensible_array.h5",
    "testdata/btree_v2.h5", "testdata/layout_v4.h5",
  ]
}

///|
test "mutated fixtures produce only HDF5 errors" {
  let rng = { state: 0x9E3779B97F4A7C15UL, }
  for path in robustness_fixtures() {
    let source = @fs.read_file_to_bytes(path)
    for _ in 0..<48 {
      exercise_bytes(mutate_fixture(source, rng))
    }
  }
}

///|
test "truncated fixtures produce only HDF5 errors" {
  for path in robustness_fixtures() {
    let source = @fs.read_file_to_bytes(path)
    let total = source.length()
    let mut length = 0
    while length < total && length < 128 {
      exercise_bytes(truncate_fixture(source, length))
      length = length + 1
    }
    let lengths : Array[Int32] = [
      total - 16,
      total - 8,
      total - 4,
      total - 2,
      total - 1,
      total / 8,
      total / 4,
      total / 2,
      total * 3 / 4,
    ]
    for selected in lengths {
      if selected >= 0 && selected < total {
        exercise_bytes(truncate_fixture(source, selected))
      }
    }
  }
}

///|
test "random buffers produce only HDF5 errors" {
  let rng = { state: 0xD1B54A32D192ED03UL, }
  for _ in 0..<256 {
    let length = rng.below(512)
    let bytes = Bytes::makei(length, _ => rng.below(256).to_byte())
    exercise_bytes(bytes)
  }
}
