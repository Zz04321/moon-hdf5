///|
fn netcdf_attribute_string(
  file : Hdf5File,
  path : String,
  name : String,
) -> String raise {
  let attributes = file.attributes(path)
  match attributes.search_by(attribute => attribute.name == name) {
    Some(index) =>
      match attributes[index].values {
        Some(Strings(values)) => {
          assert_eq(values.length(), 1)
          values[0]
        }
        _ => abort("expected NetCDF-4 string attribute " + name)
      }
    None => abort("missing NetCDF-4 attribute " + name)
  }
}

///|
test "reads a real NetCDF-4 producer file" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/netcdf4_1_7_5.h5"),
  )
  assert_eq(file.superblock.version, 2)
  assert_eq(file.superblock.offset_size, 8)

  let paths = file.walk().map(node => node.path)
  for
    path in [
      "/time", "/station", "/lat", "/lon", "/temperature", "/station_name", "/quality",
      "/metadata", "/metadata/level",
    ] {
    assert_true(paths.contains(path))
  }

  assert_eq(file.attributes("/").length(), 3)
  assert_true(
    netcdf_attribute_string(file, "/", "_NCProperties").has_prefix(
      "version=2,netcdf=4.10.1,hdf5=2.2.0",
    ),
  )
  assert_eq(
    netcdf_attribute_string(file, "/", "title"),
    "Moon HDF5 NetCDF-4 interop",
  )
  assert_eq(netcdf_attribute_string(file, "/", "institution"), "moon-hdf5")

  assert_eq(integer_values(file.read_dataset("/time")), [0L, 6L, 12L, 18L])
  assert_eq(float_values(file.read_dataset("/lat")), [-30.0, 0.0, 30.0])
  assert_eq(float_values(file.read_dataset("/lon")), [
    0.0, 90.0, 180.0, 270.0, 360.0,
  ])
  assert_eq(unsigned_values(file.read_dataset("/quality")), [0UL, 1UL, 0UL, 2UL])
  assert_eq(string_values(file.read_dataset("/station_name")), [
    "alpha", "月亮",
  ])
  assert_eq(float_values(file.read_dataset("/metadata/level")), [1000.0, 500.0])

  let temperature = file.read_dataset("/temperature")
  assert_eq(temperature.dataspace.dimensions, [4UL, 3UL, 5UL])
  let values = float_values(temperature)
  assert_eq(values.length(), 60)
  assert_eq(values[0], 250.0)
  assert_eq(values[59], 309.0)
  assert_true(temperature.layout.chunk_index is BTreeV1)
  let selected = file.read_dataset_selection(
    "/temperature",
    Hyperslab([0UL, 0UL, 0UL], [1UL, 2UL, 3UL], [1UL, 1UL, 1UL]),
  )
  match selected.values {
    Floats(values) =>
      assert_eq(values, [250.0, 251.0, 252.0, 255.0, 256.0, 257.0])
    _ => abort("expected NetCDF-4 floating-point selection")
  }

  assert_eq(
    netcdf_attribute_string(file, "/temperature", "long_name"),
    "synthetic temperature",
  )
  assert_eq(netcdf_attribute_string(file, "/temperature", "units"), "K")
  assert_eq(netcdf_attribute_string(file, "/time", "CLASS"), "DIMENSION_SCALE")
  for name in ["DIMENSION_LIST", "_Netcdf4Coordinates", "_Netcdf4Dimid"] {
    assert_true(
      file.attributes("/temperature").any(attribute => attribute.name == name),
    )
  }
}
