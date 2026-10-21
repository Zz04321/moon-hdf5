# moon-hdf5

moon-hdf5 is a pure MoonBit implementation for reading HDF5 files and creating
new HDF5 files. It has no native HDF5 dependency and supports both wasm and
native targets.

The project is under active development and implements a documented subset of
HDF5. See the [compatibility matrix](docs/compatibility.md) for exact coverage
and [audit report](docs/audit.md) for verification history.

## Features

- Reads groups, datasets, attributes, links and common HDF5 metadata versions.
- Decodes numeric, string, compound, enum, array, bitfield, opaque, reference,
  variable-length sequence, time and complex values.
- Supports compact, contiguous and chunked layouts with the common HDF5 chunk
  indexes.
- Applies deflate, shuffle and Fletcher32 filters, including per-chunk masks.
- Provides metadata-only access, hyperslab selection and ordered point
  selection.
- Creates new files with groups, attributes, hard, soft and external links,
  contiguous or chunked datasets, filters, fill values and one extensible
  dimension. Large chunk indexes use fixed-array pages or extensible-array
  super blocks and data pages.

## Quick Start

Install the [MoonBit toolchain](https://www.moonbitlang.com/download). The
native target also requires a C compiler.

```sh
git clone https://github.com/Zz04321/moon-hdf5.git
cd moon-hdf5
moon update
moon run src/cmd/main --target native -- testdata/basic_v0.h5
```

The inspector writes diagnostics to stderr and returns status 0 for success, 1
for data errors and 2 for invalid arguments. A dataset-level error does not stop
inspection of other readable datasets.

## Reading

Import the library as shown in [src/cmd/main/moon.pkg](src/cmd/main/moon.pkg):

```moonbit nocheck
///|
import {
  "Zz04321/moon-hdf5" @lib,
  "moonbitlang/x/fs",
}

///|
let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes("data.h5"))

///|
let metadata = file.read_dataset_metadata("/group/values")

///|
let dataset = file.read_dataset("/group/values")

///|
let selected = file.read_dataset_selection(
  "/matrix",
  @lib.Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 2UL]),
)

///|
let attributes = file.attributes("/group/values")

///|
let nodes = file.walk()
```

Simple values are returned as `Integers`, `UnsignedIntegers`, `Floats`,
`Strings` or `Enums`. Other datatype classes use recursive `Structured` values.
References are returned as encoded bytes and are not dereferenced.

`read_dataset_metadata` does not load dataset values. Selection reads support
compact, contiguous and chunked datasets; chunked reads decode only intersecting
allocated chunks.

`Hdf5File::parse_source` also accepts an implementation of the random-access
`Hdf5Source` interface. Superblock parsing, object headers and group navigation
use bounded reads, as do dataset values, attributes and selections
from every supported storage layout. Global heap objects are read on demand,
and chunk selections read only their index blocks and intersecting chunks.
Source file addresses use 64-bit offsets and can exceed 2 GiB.

## Writing

`Hdf5Writer` builds a complete new file in memory. It supports signed and
unsigned 8/16/32/64-bit integers, float32/64, fixed or variable-length UTF-8
strings, and structured compound, enum and fixed-array values.

```moonbit nocheck
///|
let writer = @lib.Hdf5Writer::new()

///|
writer.add_dataset(
  "/experiment/counts",
  [2UL, 3UL],
  @lib.Int64Values([1L, 2L, 3L, 4L, 5L, 6L]),
)

///|
writer.add_soft_link("/experiment", "latest", "/experiment/counts")

///|
writer.add_external_link(
  "/experiment",
  "remote_counts",
  "other-file.h5",
  "/experiment/counts",
)

///|
writer.add_extensible_dataset(
  "/experiment/series",
  [0UL, 2UL],
  [None, Some(2UL)],
  [2U, 2U],
  @lib.Int64Values([]),
  fill_value=@lib.Int64Values([-1L]),
  filters=[@lib.Shuffle, @lib.Deflate(6), @lib.Fletcher32],
)

///|
writer.append_dataset(
  "/experiment/series",
  [2UL, 2UL],
  @lib.Int64Values([10L, 11L, 20L, 21L]),
)

///|
writer.add_attribute(
  "/experiment/counts",
  "unit",
  [],
  @lib.FixedStringValues(["items"], 8),
)

///|
let bytes = writer.encode()
```

Use `add_chunked_dataset` for a finite chunked dataset. Use
`add_extensible_dataset`, `resize_dataset` and `append_dataset` for a dataset
with one unlimited dimension. `add_soft_link` attaches a compact link to an
absolute target or a target relative to the containing group; dangling links are
allowed. `add_external_link` records a target file and object path; HDF5 readers
resolve them, while this library reports external links as unresolved. The
writer rejects link messages whose encoded metadata exceeds HDF5's 16-bit message
size. `Datatype::write_integer`, `write_float`, `write_fixed_string`,
`write_enum`, `write_array` and `write_compound` construct supported structured
writer types. Large compact group and attribute metadata is split across
checksummed object-header continuation chunks, so a single object is no longer
limited to 64 KiB of messages. The writer demo includes an extensible array
large enough to exercise direct blocks, super blocks and paged data blocks, plus
an object with enough attributes to exercise header continuations.

Run the complete writer example and verify its output independently with h5py:

```sh
moon run src/cmd/write_demo --target native -- output.h5
python scripts/verify_writer_output.py output.h5
```

## Command Line

```sh
moon run src/cmd/main -- inspect testdata/basic_v0.h5
moon run src/cmd/main -- metadata testdata/basic_v0.h5 /matrix
moon run src/cmd/main -- stats testdata/basic_v0.h5 /matrix
moon run src/cmd/main -- read testdata/basic_v0.h5 /matrix 0,0 2,2 1,2
moon run src/cmd/main -- read testdata/basic_v0.h5 /matrix --points "1,2;0,1"
```

Coordinates are zero-based. Hyperslabs use comma-separated `start`, `count` and
`stride` vectors. Point selections use semicolon-separated coordinates and
preserve order and duplicates.

Focused examples are available in `src/cmd/metadata_demo`,
`src/cmd/selection_demo` and `src/cmd/write_demo`.

## Limitations

- Decoded values remain in memory even when the input uses random access.
- The writer creates complete new files and does not edit existing files.
- Dense link and dense attribute storage are not written; large compact metadata
  uses object-header continuations instead. External links can be written but
  are not resolved by the reader; external dataset storage is not supported.
- Custom variable-length fill values, additional writer filters and reference
  writing are not supported.

Version support does not imply support for every structure in that version.
Unsupported structures raise `UnsupportedFeature`; malformed data normally
raises `InvalidField` or `UnexpectedEof`. Invalid selections and writer inputs
raise `InvalidArgument`.

## Development

```sh
moon info
moon fmt
moon check
moon test --target wasm
moon test --target native
```

Run `scripts/release_check.sh` to include interface and formatting checks,
documentation generation, package contents, the writer demo and independent
h5py verification. Set `H5PY_PYTHON` when h5py is installed in a specific
environment, or `SKIP_H5PY=1` to omit only the interoperability check.

GitHub Actions CI is manual and runs checks, formatting, documentation, builds,
wasm/native tests and writer interoperability. Fixture generation is documented in
[testdata/README.md](testdata/README.md). Architecture and implementation notes
are in [docs/design.md](docs/design.md); external sources and ecosystem projects
are listed in [docs/prior-art.md](docs/prior-art.md).

## License

Apache-2.0. Zlib decompression uses
[MoonZip](https://github.com/moonbitstack/moonzip), also under Apache-2.0.
