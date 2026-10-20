# moon-hdf5

A pure MoonBit, read-only HDF5 reader. The library accepts file bytes and
exposes groups, datasets and attributes without a native HDF5 dependency.
This is an early implementation with known compatibility gaps, not a complete
HDF5 implementation. See [audit findings](docs/audit.md).

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

As checked on 2026-10-19, Mooncakes search does not find this module; use the
checkout until a registry release is verified.

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
let attributes = file.attributes("/group/values")

///|
let nodes = file.walk()
```

Values are `Integers`, `UnsignedIntegers`, `Floats`, `Strings` or `Enums`.
Enum datasets decode to member names; `Datatype.enum_members` exposes each
name and its signed or unsigned base value.
`open_dataset_metadata(path)` reads shape, datatype and layout without loading
dataset values, while `resolve(path)` locates objects without parsing dataset
metadata. See the
[inspector](src/cmd/main/main.mbt) for error handling and the
[generated interface](src/pkg.generated.mbti) for all public functions.

## Scope

| Area | Implemented subset |
| --- | --- |
| Framing | Superblocks v0–v3, user blocks, object headers v1/v2 |
| Groups | Symbol tables, compact link messages, dense groups via fractal heaps and metadata v2 B-trees; hard/soft links |
| Layout | Compact, contiguous, v1 B-tree chunks; v4 single-chunk, implicit, fixed-array, extensible-array and v2 B-tree indexes |
| Filters | Deflate (HDF5 gzip), shuffle, Fletcher32 and combinations; per-chunk skip masks |
| Values | Standard int/uint8–64, IEEE float32/64, fixed/vlen strings, enum datatypes over integer bases |
| Attributes | Inline messages v1–v3, dense attributes via fractal heaps and metadata v2 B-trees |
| Fills | Default zeros and fixed-size numeric/string fills for missing chunks and unallocated contiguous data |

Writing, other filters, external links and compound/reference types are
outside the current scope.
Undefined/disabled fill policies and custom vlen fills are rejected when fill decoding is needed. External dataset storage is unsupported.
Version support does not imply support for every structure in that version.
Nonstandard numeric layouts are explicitly rejected. Fractal heaps support
I/O filter pipelines and managed, tiny, direct-huge and indirect-huge objects.
Empty/null contiguous datasets return empty values. Object-header checksums
and other supported metadata checksums are verified with HDF5's lookup3
algorithm, covering superblocks, object-header chunks and continuations,
v2 B-trees, fractal heaps, fixed arrays and extensible arrays.
Inputs and decoded datasets reside in memory; this is not a streaming reader.

## Development

`src/` contains the library and its tests; `src/cmd/main/` contains the CLI.
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

CI runs manually through GitHub Actions → CI → Run workflow; it runs `moon
check`, builds both wasm and native targets, and runs the tests for both.
Ordinary tests use committed fixtures and do not need Python. See [fixture
generation](testdata/README.md),
[design](docs/design.md) and [ecosystem comparison and sources](docs/prior-art.md).

Apache-2.0; see [LICENSE](LICENSE). Zlib decompression uses the pure MoonBit
[MoonZip](https://github.com/moonbitstack/moonzip) library (Apache-2.0).
