# moon-hdf5

A pure MoonBit HDF5 reader with an initial writer. The library accepts file
bytes and exposes groups, datasets and attributes without a native HDF5
dependency. It can also create new files containing nested groups, contiguous
basic datasets and inline attributes. This is an early implementation with
known compatibility gaps, not a complete HDF5 implementation. See
[audit findings](docs/audit.md).

## Run from source

Requires the [MoonBit toolchain](https://www.moonbitlang.com/download);
the native backend also needs a C compiler.

```sh
git clone https://github.com/Zz04321/moon-hdf5.git
cd moon-hdf5
moon update
moon run src/cmd/main --target native -- testdata/basic_v0.h5
```

Replace the final argument with your file path. Output `int[10] 0` means ten
integers with first value zero. The inspector loads complete datasets. Diagnostics
go to stderr; exit status is 0 for success, 1 for read/parse/unsupported-data
errors (including attributes), and 2 for invalid arguments. Other readable
datasets are still inspected after a dataset-level error.

## Library API

Within this checkout, use imports as in [src/cmd/main/moon.pkg](src/cmd/main/moon.pkg):

```moonbit nocheck
///|
import {
  "Zz04321/moon-hdf5" @lib,
  "moonbitlang/x/fs",
}
```

In a function that handles or propagates errors:

```moonbit nocheck
///|
let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes("data.h5"))

///|
let dataset = file.open_dataset("/group/values")

///|
let shape = dataset.dataspace.dimensions

///|
let metadata = file.open_dataset_metadata("/group/values")

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
`Strings` or `Enums`. Other datatype classes use `Structured`, whose recursive
`DatasetValue` entries represent time, bitfield, opaque, compound, reference,
variable-length sequence, array and complex values. References expose their
encoded bytes; the reader does not dereference the target object or region.
Enum datasets decode to member names; `Datatype.enum_members` exposes each name
and its signed or unsigned base value.
`open_dataset_metadata(path)` reads shape, datatype and layout without loading
dataset values, while `resolve(path)` locates objects without parsing dataset
metadata. `read_dataset_selection` supports strided hyperslabs and ordered point
selections for compact, contiguous and chunked datasets. Chunked reads decode
only intersecting allocated chunks and preserve fill values for missing chunks.
See the [inspector](src/cmd/main/main.mbt) for error handling and the
[generated interface](src/pkg.generated.mbti) for all public functions.

## Writing

`Hdf5Writer` builds a new file in memory. It supports nested groups, contiguous
or fixed-array chunked `Int64`, `UInt64`, `Float64` and fixed-width UTF-8 string
datasets, scalar or simple dataspaces, and inline attributes with the same
types. Chunked datasets can omit fill-only chunks and apply shuffle, deflate
and Fletcher32 filters.

```moonbit nocheck
///|
let writer = @lib.Hdf5Writer::new()

///|
writer.add_group("/experiment")

///|
writer.add_dataset(
  "/experiment/counts",
  [2UL, 3UL],
  @lib.Int64Values([1L, 2L, 3L, 4L, 5L, 6L]),
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

For chunked data, call `add_chunked_dataset` with the dataset and chunk shapes.
Its optional `fill_value` must contain exactly one value of the dataset type;
its optional `filters` use HDF5 pipeline order, such as
`[Shuffle, Deflate(6), Fletcher32]`. This non-paged fixed-array writer currently
limits each dataset to 1024 chunks.

Run `moon run src/cmd/write_demo --target native -- output.h5` for a complete
file-writing example. The output is readable by this library and h5py.

## Scope

| Area | Implemented subset |
| --- | --- |
| Framing | Superblocks v0–v3, user blocks, object headers v1/v2 |
| Groups | Symbol tables, compact link messages, dense groups via fractal heaps and metadata v2 B-trees; hard/soft links |
| Layout | Compact, contiguous, v1 B-tree chunks; v4 single-chunk, implicit, fixed-array, extensible-array and v2 B-tree indexes |
| Filters | Deflate (HDF5 gzip), shuffle, Fletcher32 and combinations; per-chunk skip masks |
| Values | Standard int/uint8–64, IEEE float32/64, time, fixed/vlen strings, bitfield, opaque, compound, reference, enum, vlen sequence, array and complex datatypes |
| Attributes | Inline messages v1–v3, dense attributes via fractal heaps and metadata v2 B-trees |
| Fills | Default zeros and fixed-size fills for missing chunks and unallocated contiguous data |
| Selections | Strided hyperslabs and ordered points, including duplicates, across supported layouts and chunk indexes |
| Writing | New files with nested groups, contiguous/fixed-array chunked basic datasets, fills, shuffle/deflate/Fletcher32 and inline attributes |

Extensible datasets, variable-length writer values, additional filters and
external links are outside the current scope.
Undefined/disabled fill policies and custom variable-length fills are rejected
when fill decoding is needed. External dataset storage is unsupported.
Version support does not imply support for every structure in that version.
Nonstandard numeric layouts are explicitly rejected. Fractal heaps support
I/O filter pipelines and managed, tiny, direct-huge and indirect-huge objects.
Empty/null contiguous datasets return empty values. Object-header checksums
and other supported metadata checksums are verified with HDF5's lookup3
algorithm, covering superblocks, object-header chunks and continuations,
v2 B-trees, fractal heaps, fixed arrays and extensible arrays.
Inputs and decoded datasets reside in memory; this is not a streaming reader.

## Development

`src/` contains the library and its tests; `src/cmd/main/` contains the
inspector and `src/cmd/write_demo/` contains a writer example.
`moon.mod` sets the source root, preserving the library import path
`Zz04321/moon-hdf5`. Documentation, fixture generators and test data live in
`docs/`, `scripts/` and `testdata/`. Run the following commands from the repository root.

```sh
moon check
moon test
moon test --target native
moon info
moon fmt
```

To verify native writer output independently, run the writer example followed
by `python scripts/verify_writer_output.py output.h5` in an environment with
h5py and NumPy.

CI runs manually through GitHub Actions → CI → Run workflow; it runs `moon
check`, builds both wasm and native targets, and runs the tests for both.
Ordinary tests use committed fixtures and do not need Python. See [fixture
generation](testdata/README.md),
[design](docs/design.md) and [ecosystem comparison and sources](docs/prior-art.md).

Apache-2.0; see [LICENSE](LICENSE). Zlib decompression uses the pure MoonBit
[MoonZip](https://github.com/moonbitstack/moonzip) library (Apache-2.0).
