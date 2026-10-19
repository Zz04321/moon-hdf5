# Ecosystem check and references

Checked 2026-10-19. No published, mature MoonBit HDF5 file parser with highly
overlapping scope was found. This does not prove that no related work exists.

```sh
moon search hdf5 --limit 50
moon search hdf --limit 50
curl 'https://mooncakes.io/api-new/v0/search?kw=hdf5'
```

CLI queries returned no modules; the registry returned `[]`. GitHub repository
queries `hdf5 moonbit` and `hdf5 language:moonbit` found only this repository
after publication. Authenticated code search also found related implementations:

| Project | Relationship |
| --- | --- |
| [cf_ocean](https://github.com/hrwqe/moonbit-cf-ocean) | NetCDF Classic/CDF-2; excludes NetCDF-4/HDF5 |
| [zarr](https://github.com/zlhahaha/zarr) | Zarr v2/v3, a different storage format |
| [MoonMatBridge](https://github.com/WeiR-h/moonmatbridge) | MAT Level 5; excludes HDF5 MAT v7.3 |
| [BioSeqs rhdf5](https://github.com/paipai-Studio/BioSeqs/blob/main/src/rhdf5.mbt) | Inspected module models data in memory, without HDF5 binary parsing |

Research followed the package-lookup workflow in
[osc2026-guide](https://github.com/Milky2018/osc2026-guide/blob/main/SKILL.md).
This project's contribution is parsing HDF5 container bytes in MoonBit.

## Implementation sources

- [HDF5 specification](https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html): on-disk format.
- [pyfive](https://github.com/NCAS-CMS/pyfive), BSD-3-Clause: layout cross-checks.
- [hdf5-rust](https://github.com/aldanor/hdf5-rust), MIT/Apache-2.0: API and scope reference.
- [HDF5 fill decoder](https://github.com/HDFGroup/hdf5/blob/develop/src/H5Ofill.c): fill-message flags and versions.
- [HDF5 fixed arrays](https://github.com/HDFGroup/hdf5/blob/develop/src/H5FAcache.c): page layout; `H5FA.c` and `H5VMprivate.h` were consulted for page offsets and bitmap bit order.
- [HDF5 extensible arrays](https://github.com/HDFGroup/hdf5/blob/develop/src/H5EAcache.c): block layout; `H5EAhdr.c`, `H5EA.c`, `H5EApkg.h`, `H5Dearray.c` and `H5Olayout.c` were consulted for block sizing, offsets, dimension ordering and creation parameters.
- [HDF5 v2 B-trees](https://github.com/HDFGroup/hdf5/blob/develop/src/H5B2cache.c): node layout; `H5B2hdr.c`, `H5B2pkg.h` and `H5Dbtree2.c` were consulted for pointer widths, capacities and chunk record encoding.

MoonBit code is original, with AI assistance; reference source code was not copied.
Fixtures are generated h5py data; see [provenance and regeneration](../testdata/README.md).

## Runtime compression dependency

`moonbitstack/moonzip@0.3.0` provides zlib/DEFLATE decoding with checksum and
output-limit options; its transitive dependency is `moonbitstack/moonbase@0.4.0`.
Both use Apache-2.0. They are package dependencies, not copied source.
Fletcher32 layout and checksum byte order were cross-checked against HDF5's
`H5Zfletcher32.c` and `H5checksum.c`; our implementation is verified with h5py
fixtures, including odd byte counts and deliberately corrupted chunks.
