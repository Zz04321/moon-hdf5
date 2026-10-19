# Design

The library operates on in-memory `Bytes`; filesystem access belongs to the caller.

| Layer | Files | Responsibility |
| --- | --- | --- |
| Framing | reader, superblock, object_header | Byte reads, addresses, messages |
| Navigation | heap, symbol_table, link, hdf5_file | Groups, links and paths |
| Metadata | dataspace, datatype, data_layout | Shapes, types and storage |
| Values | dataset, chunked, fill_value, global_heap, attribute | Value decoding |

Addresses are resolved relative to the superblock base. Modern group link-info
messages are checked for allocated heap/index addresses before returning compact
links. Chunked reads initialize the output with fill bytes before copying allocated
chunks. The low-level `read_chunked_payload` defaults to zeros; the high-level
reader supplies parsed fill values.

Recognized unsupported features raise `UnsupportedFeature`; malformed fields
normally raise `InvalidField` or `UnexpectedEof`. Unsupported attribute values
can be represented as `None`. Detection is incomplete: not all unsupported
inputs are rejected, and metadata checksums are not verified. See [audit.md](audit.md).

Numeric metadata is checked before decoding: integers require full-width bits
with zero offset; floats require standard IEEE binary32/binary64 fields.
Attribute-info heap/index addresses determine dense storage independently of
creation-order tracking. Unsupported numeric layouts raise an error, including
when used in attributes; the inspector prints attribute errors with their path.

Public interfaces are generated in `pkg.generated.mbti`. Tests use committed
h5py fixtures and synthetic messages. Local validation covers wasm and native;
manual CI currently runs only default-target checks/tests. Regeneration
instructions are in [testdata/README.md](../testdata/README.md).
