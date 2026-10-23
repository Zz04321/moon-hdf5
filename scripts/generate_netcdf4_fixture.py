from pathlib import Path

import netCDF4
import numpy as np


def populate(root: netCDF4.Dataset) -> np.ndarray:
    root.title = "Moon HDF5 NetCDF-4 interop"
    root.institution = "moon-hdf5"
    root.createDimension("time", 4)
    root.createDimension("lat", 3)
    root.createDimension("lon", 5)
    root.createDimension("station", 2)

    time = root.createVariable("time", "i4", ("time",))
    time.units = "hours since 2026-01-01 00:00:00"
    time[:] = np.array([0, 6, 12, 18], dtype=np.int32)

    lat = root.createVariable("lat", "f4", ("lat",))
    lat.units = "degrees_north"
    lat[:] = np.array([-30.0, 0.0, 30.0], dtype=np.float32)

    lon = root.createVariable("lon", "f4", ("lon",))
    lon.units = "degrees_east"
    lon[:] = np.array([0.0, 90.0, 180.0, 270.0, 360.0], dtype=np.float32)

    temperature = root.createVariable(
        "temperature",
        "f4",
        ("time", "lat", "lon"),
        zlib=True,
        complevel=6,
        shuffle=True,
        chunksizes=(2, 2, 3),
    )
    temperature.units = "K"
    temperature.long_name = "synthetic temperature"
    values = np.arange(60, dtype=np.float32).reshape(4, 3, 5) + 250.0
    temperature[...] = values

    station = root.createVariable("station_name", str, ("station",))
    station[0] = "alpha"
    station[1] = "月亮"

    quality = root.createVariable("quality", "u1", ("time",))
    quality.long_name = "quality flag"
    quality[:] = np.array([0, 1, 0, 2], dtype=np.uint8)

    metadata = root.createGroup("metadata")
    metadata.description = "producer metadata"
    metadata.createDimension("level", 2)
    level = metadata.createVariable("level", "f8", ("level",))
    level[:] = np.array([1000.0, 500.0], dtype=np.float64)
    return values


def verify(destination: Path, expected_temperature: np.ndarray) -> None:
    with netCDF4.Dataset(destination, "r") as root:
        assert root.data_model == "NETCDF4"
        assert root.title == "Moon HDF5 NetCDF-4 interop"
        np.testing.assert_array_equal(root["time"][:], [0, 6, 12, 18])
        np.testing.assert_array_equal(root["lat"][:], [-30.0, 0.0, 30.0])
        np.testing.assert_array_equal(root["lon"][:], [0.0, 90.0, 180.0, 270.0, 360.0])
        np.testing.assert_array_equal(root["temperature"][:], expected_temperature)
        assert root["station_name"][:].tolist() == ["alpha", "月亮"]
        np.testing.assert_array_equal(root["quality"][:], [0, 1, 0, 2])
        np.testing.assert_array_equal(root["metadata/level"][:], [1000.0, 500.0])


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "testdata" / "netcdf4_1_7_5.h5"
    with netCDF4.Dataset(destination, "w", format="NETCDF4") as root:
        expected_temperature = populate(root)
    verify(destination, expected_temperature)
    print(
        f"Verified {destination.name} with netCDF4 {netCDF4.__version__}, "
        f"libnetcdf {netCDF4.__netcdf4libversion__} and "
        f"HDF5 {netCDF4.__hdf5libversion__}"
    )
