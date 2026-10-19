# Prior-art and duplicate check

Check date: 2026-10-19.

## Mooncakes.io

The official registry search API and the MoonBit CLI were both queried:

```bash
moon search hdf5 --limit 50   # No modules found.
moon search hdf --limit 50    # No modules found.
curl 'https://mooncakes.io/api-new/v0/search?kw=hdf5'  # []
curl 'https://mooncakes.io/api-new/v0/search?kw=hdf'   # []
```

`moon search h5` returns only a WeChat/Alipay payment SDK, which is unrelated.
There is currently no MoonBit package whose name, description or keywords
advertise an HDF5 reader or writer.

GitHub repository search on the same date also returned zero results for:

- `hdf5 moonbit`
- `hdf5 language:moonbit`
- `moonbit hdf5`
- `hdf5 moon`

GitHub's code search is not available without an authenticated session, so the
repository-level search cannot prove the absence of an HDF5 implementation
hidden inside an unrelated monorepo. The Mooncakes registry remains the primary
evidence because accepted contest projects are expected to be published there.

## Related MoonBit packages

Two packages are adjacent but do not overlap:

- `hrwqe/cf_ocean` implements NetCDF Classic (CDF-1) and 64-bit Offset (CDF-2).
  Its README explicitly states: "NetCDF-4/HDF5 containers are not implemented".
- `zlhahaha/zarr` implements Zarr v2 arrays and chunk metadata, not the HDF5
  container format, object headers, symbol tables or HDF5 chunk B-trees.

`moon-hdf5` therefore fills a separate gap: the HDF5 container format itself.
It can serve as the metadata layer for future NetCDF-4 and HDF5-backed Zarr
work, while remaining independent of either project.

## References used for format verification

- HDF5 File Format Specification:
  <https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html>
- `pyfive` (BSD-3-Clause): used only to cross-check on-disk layout details.
- `hdf5-rust`: used only to assess layout complexity and feature boundaries.

No source code was copied from these projects. Fixtures were generated with
h5py 3.16.0 and are committed under `testdata/` so CI does not depend on Python
or a native HDF5 installation.
