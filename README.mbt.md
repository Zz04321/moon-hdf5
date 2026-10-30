# moon-hdf5

Pure MoonBit HDF5 reader and new-file writer for the HDF5 1.14.6 common-file
profile. It has no native HDF5 dependency and runs on wasm and native targets.

moon-hdf5 implements a documented subset of HDF5. The
[compatibility matrix](docs/compatibility.md) is the v1.0 delivery boundary;
[design notes](docs/design.md) describe the implementation and
[audit notes](docs/audit.md) record verification. Public APIs are listed in
[`src/pkg.generated.mbti`](src/pkg.generated.mbti).

## Status

- `Hdf5File` reads existing files without modification.
- `Hdf5Writer` creates complete new files in memory.
- `Hdf5Resolver` follows registered external links and reads resolver-backed
  external raw data.
- In-place editing of an existing file is not supported.

## Capabilities

- Superblocks v0–v3, 4- and 8-byte addresses, compact and dense groups and
  attributes, hard/soft/external links, and every named hard-link path.
- Compact, contiguous and chunked datasets; v1 B-tree, single-chunk, implicit,
  fixed-array, extensible-array and v2 B-tree indexes.
- Deflate, shuffle, Fletcher32, integer scale-offset and f32/f64 decimal
  scale-offset, including per-chunk filter masks.
- Signed and unsigned 1–8-byte integers in either byte order, IEEE
  float16/32/64, normalized binary floats with implied MSB, strings, compound,
  enum, array, time, bitfield, opaque, sequence, reference and complex values.
- Hyperslab and ordered-point reads, metadata-only reads, bounded-memory chunk
  iteration, object-reference resolution and dataset-region-reference reads.
- SOHM reading and optional writing: shared attributes, datatype/dataspace
  fields, fills and named committed datatypes.
- New-file output for groups, datasets, attributes, links, committed datatypes,
  filters, fills, references, contiguous external raw-data slots and one or more
  unlimited dataset dimensions.

## Quick Start

Install the [MoonBit toolchain](https://www.moonbitlang.com/download); the native
target also needs a C compiler.

```sh
git clone https://github.com/Zz04321/moon-hdf5.git
cd moon-hdf5
moon update
moon run src/cmd/main --target native -- testdata/basic_v0.h5
```

## Reading

Imports for the examples below are in
[`src/cmd/main/moon.pkg`](src/cmd/main/moon.pkg).

```moonbit nocheck
///|
let bytes = @fs.read_file_to_bytes("data.h5")
let file = @lib.Hdf5File::parse(bytes)

let metadata = file.read_dataset_metadata("/group/values")
let values = file.read_dataset("/group/values")
let slice = file.read_dataset_selection(
  "/matrix",
  @lib.Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 2UL]),
)
let attributes = file.attributes("/group/values")
let nodes = file.walk()
```

`read_dataset_metadata` does not load values. Selections support compact,
contiguous and chunked storage, and only intersecting chunks are decoded.
`chunk_iterator` returns edge-clipped tiles with memory bounded by one tile plus
the intersecting storage chunk.

```moonbit nocheck
///|
let chunks = file.chunk_iterator("/matrix", [1UL, 2UL])
while chunks.next() is Some(chunk) {
  chunk.offset |> ignore
  chunk.values |> ignore
}

///|
let resolver = @lib.Hdf5Resolver::new(bytes)
resolver.add_source("other-file.h5", @fs.read_file_to_bytes("other-file.h5"))
let remote = resolver.read_dataset("/experiment/remote_counts")
```

Simple values use `Integers`, `UnsignedIntegers`, `Floats`, `Strings` or
`Enums`; other datatypes use recursive `Structured` values. References are
returned as encoded bytes and can be resolved with
`resolve_object_reference`, `resolve_region_reference` or
`read_region_reference`. Region targets must be datasets in the same file.

`Hdf5Resolver` follows hard, soft and external links across registered
`Hdf5Source` values, rejects cycles and reports unregistered targets. It also
reads contiguous, unfiltered external raw-data slots. `Hdf5File` intentionally
leaves external targets and external raw data unresolved.

## Writing

```moonbit nocheck
///|
let writer = @lib.Hdf5Writer::new()
writer.set_attribute_sharing(true)
writer.set_fill_sharing(true)

let record = @lib.Datatype::write_compound([
  ("id", @lib.Datatype::write_integer(true, 4)),
  ("value", @lib.Datatype::write_float(8)),
])
writer.add_committed_datatype("/types/record", record)
writer.add_dataset(
  "/experiment/records",
  [2UL],
  @lib.StructuredValues(record, [
    @lib.Compound([
      { name: "id", value: @lib.SignedInteger(1L) },
      { name: "value", value: @lib.FloatingPoint(1.25) },
    ]),
    @lib.Compound([
      { name: "id", value: @lib.SignedInteger(2L) },
      { name: "value", value: @lib.FloatingPoint(-3.5) },
    ]),
  ]),
)
writer.add_attribute(
  "/experiment/records",
  "unit",
  [],
  @lib.FixedStringValues(["items"], 8),
)
writer.add_chunked_dataset(
  "/experiment/codes",
  [6UL],
  [3U],
  @lib.Int64Values([10L, 11L, 12L, 20L, 21L, 22L]),
  fill_value=@lib.Int64Values([-1L]),
  filters=[@lib.Shuffle, @lib.Deflate(6), @lib.Fletcher32],
)
writer.add_extensible_dataset(
  "/experiment/series",
  [0UL, 2UL],
  [None, Some(2UL)],
  [2U, 2U],
  @lib.Int64Values([]),
  fill_value=@lib.Int64Values([-1L]),
)
writer.append_dataset(
  "/experiment/series",
  [2UL, 2UL],
  @lib.Int64Values([1L, 2L, 3L, 4L]),
)
writer.add_soft_link("/experiment", "latest", "/experiment/series")

let output = writer.encode()
```

- `add_dataset` writes contiguous storage. `add_chunked_dataset`,
  `add_extensible_dataset` and `resize_dataset` cover finite and multi-unlimited
  chunked storage. One unlimited dimension uses an extensible-array index;
  multiple unlimited dimensions use a v2 B-tree. `append_dataset` appends along
  a single unlimited dimension; use `resize_dataset` for multiple unlimited
  dimensions.
- `add_external_dataset` stores raw bytes in caller-managed external files. Slot
  sizes must match the dataset byte count; the writer does not create those
  files.
- Dense link and attribute collections support 9–65,535 records with encoded
  heap objects up to 65,535 bytes. Larger sets remain compact and use OCHK
  continuation chunks.
- Whole attributes and fills up to 65,535 bytes use one SOHM table and heap.
  Committed datatypes use wrapper version 3 storage type 2 and are referenced by
  datasets and attributes with matching encoded datatypes.
- Variable-length strings and sequences share a global heap. Sequence bases must
  be fixed-size and cannot contain region references.

Run the larger example and verify it independently with h5py:

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
`stride`; point lists use semicolons. Exit status is 0 for success, 1 for data
errors and 2 for invalid arguments.

## Limitations

- Whole-dataset and single-selection results are memory resident; use
  `chunk_iterator` for large reads.
- The writer creates new files only; it does not edit existing files.
- Dense output uses one direct FHDB block and one leaf per index. Indirect or
  huge heap blocks and a free-space manager are not written.
- One SOHM table and heap are emitted. Multiple SOHM tables or heaps are not
  supported.
- External raw data is limited to contiguous, unfiltered fixed-size storage;
  remote URLs, filtered/chunked external layouts and automatic external-file
  creation are unsupported.
- Numeric storage wider than 8 bytes and binary-float normalizations other than
  implied MSB are unsupported.
- Variable-length datasets cannot use filters. Custom variable-length sequence
  fills and region references inside sequences are rejected.
- Reference support covers legacy object and dataset-region forms under 4- and
  8-byte addresses. Version-4 encoded references are unsupported.
- `append_dataset` requires exactly one unlimited dimension. Multi-unlimited
  datasets are resized with `resize_dataset`.

Unsupported structures raise `UnsupportedFeature`; malformed input raises
`InvalidField` or `UnexpectedEof`; invalid caller input raises
`InvalidArgument`.

## Development

```sh
moon info
moon fmt
moon check
moon test --target wasm
moon test --target native
H5PY_PYTHON=/path/to/python ./scripts/release_check.sh
```

`scripts/release_check.sh` runs formatting, documentation and package checks,
both test targets, the writer demo and independent h5py verification. The
baseline is h5py 3.13.0 with HDF5 1.14.6. Set `SKIP_H5PY=1` to omit Python
interoperability checks. Fixture regeneration is documented in
[testdata/README.md](testdata/README.md).

## License

Apache-2.0. Zlib decompression uses
[MoonZip](https://github.com/moonbitstack/moonzip), also Apache-2.0.
