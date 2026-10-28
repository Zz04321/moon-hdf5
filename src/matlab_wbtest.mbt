///|
fn matlab_attribute(
  attributes : Array[Attribute],
  name : String,
) -> Attribute raise Hdf5Error {
  match attributes.search_by(attribute => attribute.name == name) {
    Some(index) => attributes[index]
    None => raise InvalidField("missing MATLAB fixture attribute: " + name)
  }
}

///|
test "reads a MATLAB 7.3 compatible producer fixture" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/matlab73_compatible.mat"),
  )
  let expected_paths = [
    "/counts", "/empty", "/enabled", "/label", "/record", "/record/count", "/record/samples",
    "/temperature",
  ]
  for path in expected_paths {
    file.resolve(path) |> ignore
  }

  let temperature = file.read_dataset("/temperature")
  assert_eq(temperature.dataspace.dimensions, [2UL, 2UL])
  assert_eq(temperature.datatype.size, 8)
  match temperature.values {
    Floats(values) => assert_eq(values, [21.5, 23.25, 22.0, -4.5])
    _ => abort("expected MATLAB double values")
  }
  let slab = file.read_dataset_selection(
    "/temperature",
    Hyperslab([0UL, 0UL], [2UL, 1UL], [1UL, 1UL]),
  )
  match slab.values {
    Floats(values) => assert_eq(values, [21.5, 22.0])
    _ => abort("expected MATLAB double hyperslab")
  }
  let temperature_attributes = file.attributes("/temperature")
  let temperature_class = matlab_attribute(
    temperature_attributes, "MATLAB_class",
  )
  match temperature_class.values {
    Some(Strings(values)) => assert_eq(values, ["double"])
    _ => abort("expected MATLAB_class=double")
  }

  match file.read_dataset("/counts").values {
    Integers(values) => assert_eq(values, [0L, 3L, 1L, 4L, 2L, 5L])
    _ => abort("expected MATLAB int32 values")
  }
  match file.read_dataset("/enabled").values {
    UnsignedIntegers(values) => assert_eq(values, [1UL, 0UL, 0UL, 1UL])
    _ => abort("expected MATLAB logical storage values")
  }
  match file.read_dataset("/label").values {
    UnsignedIntegers(values) => assert_eq(values, [28201UL, 24230UL])
    _ => abort("expected MATLAB char storage values")
  }
  let label_class = matlab_attribute(file.attributes("/label"), "MATLAB_class")
  match label_class.values {
    Some(Strings(values)) => assert_eq(values, ["char"])
    _ => abort("expected MATLAB_class=char")
  }

  let record_attributes = file.attributes("/record")
  let record_class = matlab_attribute(record_attributes, "MATLAB_class")
  match record_class.values {
    Some(Strings(values)) => assert_eq(values, ["struct"])
    _ => abort("expected MATLAB_class=struct")
  }
  let fields = matlab_attribute(record_attributes, "MATLAB_fields")
  assert_true(fields.values is Some(Structured(values)) && values.length() == 2)
  match file.read_dataset("/record/count").values {
    UnsignedIntegers(values) => assert_eq(values, [42UL])
    _ => abort("expected MATLAB struct scalar field")
  }
  match file.read_dataset("/record/samples").values {
    UnsignedIntegers(values) => assert_eq(values, [1UL, 2UL, 3UL])
    _ => abort("expected MATLAB struct vector field")
  }

  let empty = matlab_attribute(file.attributes("/empty"), "MATLAB_empty")
  match empty.values {
    Some(UnsignedIntegers(values)) => assert_eq(values, [1UL])
    _ => abort("expected MATLAB_empty marker")
  }
  match file.read_dataset("/empty").values {
    UnsignedIntegers(values) => assert_eq(values, [1UL, 0UL])
    _ => abort("expected MATLAB empty-array dimension storage")
  }
}
