import sys
from pathlib import Path

import h5py
import numpy as np


def verify(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        assert handle.attrs["title"] == b"Moon HDF5 NetCDF-4 interop"
        assert handle.attrs["institution"] == b"moon-hdf5"
        assert handle.attrs["_NCProperties"].decode("utf-8").startswith(
            "version=2,netcdf=4.10.1,hdf5=2.2.0"
        )
        np.testing.assert_array_equal(handle["time"][:], [0, 6, 12, 18])
        np.testing.assert_array_equal(handle["lat"][:], [-30.0, 0.0, 30.0])
        np.testing.assert_array_equal(
            handle["lon"][:], [0.0, 90.0, 180.0, 270.0, 360.0]
        )
        temperature = handle["temperature"]
        assert temperature.shape == (4, 3, 5)
        assert temperature.chunks == (2, 2, 3)
        assert temperature.compression == "gzip"
        assert temperature.shuffle
        expected = np.arange(60, dtype=np.float32).reshape(4, 3, 5) + 250.0
        np.testing.assert_array_equal(temperature[:], expected)
        assert handle["station_name"].asstr()[:].tolist() == ["alpha", "月亮"]
        np.testing.assert_array_equal(handle["quality"][:], [0, 1, 0, 2])
        np.testing.assert_array_equal(handle["metadata/level"][:], [1000.0, 500.0])
    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_netcdf4_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
