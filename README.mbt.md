# moon-hdf5

A read-only [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file reader written
in pure MoonBit. It is intended for scientific and ML data pipelines that need
to inspect `.h5` files without a native HDF5 dependency.

## Features

- HDF5 superblocks v0, v1, v2 and v3, including files with a user block.
- Version 1 and version 2 object headers, including continuation blocks.
- Version 1 groups backed by symbol tables and version 2 groups backed by link
  messages.
- Hard links and soft links; external links are detected and reported.
- Datasets with compact, contiguous and unfiltered chunked layout, including
  edge chunks in version 1 chunk B-trees.
- Fixed-point integers (8/16/32/64-bit, signed and unsigned, both byte orders),
  IEEE float32/float64, fixed-length strings and variable-length strings stored
  in the global heap.
- Attribute messages v1, v2 and v3 for numeric, fixed-string and vlen-string
  values.
- Path-based lookup, object-tree walking and a command-line inspector.

## Relationship to existing MoonBit packages

Mooncakes currently has no HDF5 implementation. The adjacent
[`hrwqe/cf_ocean`](https://github.com/hrwqe/moonbit-cf-ocean) package covers
NetCDF Classic/CDF-2 and explicitly excludes NetCDF-4/HDF5 containers, while
[`zlhahaha/zarr`](https://github.com/zlhahaha/zarr) covers Zarr v2 arrays.
`moon-hdf5` targets the HDF5 container layer itself and can complement both.
See `docs/prior-art.md` for the reproducible duplicate-check record.

## Quick start

Add the module to a MoonBit project with `moon add Zz04321/moon-hdf5`, then use
the high-level API:

```moonbit nocheck
let bytes = @fs.read_file_to_bytes("data.h5")
let file = @lib.Hdf5File::parse(bytes)

for node in file.walk() {
  println(node.path)
}

let dataset = file.open_dataset("/group/values")
match dataset.values {
  Integers(values) => println(values.length().to_string())
  Floats(values) => println(values.length().to_string())
  Strings(values) => println(values.length().to_string())
  UnsignedIntegers(values) => println(values.length().to_string())
}

let attributes = file.attributes("/group/values")
for attribute in attributes {
  println(attribute.name)
}
```

The low-level parsers (`parse_superblock`, `parse_object_header`,
`read_dataset`, `read_attributes`, and so on) are also public for callers that
need direct control.

## Command-line inspector

```bash
moon run cmd/main --target native -- testdata/basic_v0.h5
```

Example output:

```text
testdata/basic_v0.h5 (superblock v0)
  attr title = string[1] hello
/group  group
  attr answer = int[1] 42
/group/values  dataset
  int[10] 0
  attr unit = string[1] count
/matrix  dataset
  float[6] 0
```

## Supported and unsupported data

Supported today:

| Area | Coverage |
| --- | --- |
| Superblock | v0/v1 (legacy) and v2/v3 (latest), user blocks |
| Object header | v1 and v2, continuation blocks |
| Groups | symbol table B-trees and compact link messages |
| Dataset layout | compact, contiguous, unfiltered chunked (v1 chunk B-tree) |
| Datatypes | int8/16/32/64, uint8/16/32/64, float32/64, fixed and vlen strings |
| Attributes | messages v1/v2/v3, scalar and small array values |
| Navigation | root/group traversal, absolute paths, soft links |

Not supported yet (reported as `UnsupportedFeature`, never silently decoded):

- Filtered chunked datasets, including gzip/deflate, shuffle and fletcher32.
- Compound, enum, array, bitfield, opaque and reference datatypes.
- Dense group and attribute storage, and modern chunk indexes.
- External links and writing HDF5 files.

## Project layout

- `reader.mbt` - bounds-checked byte cursor.
- `superblock.mbt`, `object_header.mbt` - file and object framing.
- `heap.mbt`, `symbol_table.mbt`, `link.mbt` - groups and links.
- `dataspace.mbt`, `datatype.mbt`, `data_layout.mbt` - metadata messages.
- `dataset.mbt`, `chunked.mbt`, `global_heap.mbt` - data decoding.
- `attribute.mbt` - attribute decoding.
- `hdf5_file.mbt` - high-level file/path API.
- `cmd/main` - runnable inspector.
- `docs/design.md` - implementation and error-policy notes.
- `docs/prior-art.md` - duplicate-check evidence and related packages.

## Testing

```bash
moon check
moon test
```

The test suite runs against small fixtures generated once with h5py 3.16 and
committed under `testdata/`. CI therefore does not require Python or an HDF5
installation. Fixtures cover legacy and latest superblocks, user blocks,
symbol-table and link-message groups, integer/float/string datasets, scalar
values, chunked edge cases, attributes and soft links.

Group-storage regression fixtures distinguish empty and compact groups from
dense groups, which raise `UnsupportedFeature` during link listing, traversal
or lookup through the group. See `testdata/README.md` for the h5py generator
and expected contents.

## Implementation notes

The implementation follows the public
[HDF5 File Format Specification](https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html).
The pure-Python `pyfive` reader and `hdf5-rust` were used only to cross-check
on-disk layouts; no source code was copied from them. All MoonBit code in this
repository is original.

## License

Apache-2.0. See `LICENSE`.
