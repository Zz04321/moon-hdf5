from pathlib import Path

import h5py
import numpy as np


def require_hdf5_1_14_6() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            "scaleoffset fixture requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )


def generate(destination: Path) -> None:
    require_hdf5_1_14_6()
    expected = {}
    with h5py.File(destination, "w", libver="latest") as handle:
        dataset = handle.create_dataset(
            "int32_auto",
            shape=(20,),
            dtype="<i4",
            chunks=(10,),
            fillvalue=-7,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        values = np.arange(-50, -30, dtype="<i4")
        dataset[0:10] = values[:10]
        expected["int32_auto"] = np.concatenate(
            (values[:10], np.full((10,), -7, dtype="<i4"))
        )

        values = np.arange(100, 116, dtype="<i4").reshape(4, 4)
        handle.create_dataset(
            "int32_fixed",
            data=values,
            chunks=(2, 4),
            fillvalue=127,
            compression="scaleoffset",
            scaleoffset=6,
            track_times=False,
        )
        expected["int32_fixed"] = values

        values = np.array([10, -7, 12, 13], dtype="<i4")
        handle.create_dataset(
            "int32_mixed_fill",
            data=values,
            chunks=(4,),
            fillvalue=-7,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        expected["int32_mixed_fill"] = values

        values = np.full((4,), -7, dtype="<i4")
        handle.create_dataset(
            "int32_all_fill",
            data=values,
            chunks=(4,),
            fillvalue=-7,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        expected["int32_all_fill"] = values

        values = np.array([-5, -4, -3, -2], dtype="<i4")
        handle.create_dataset(
            "int32_full",
            data=values,
            chunks=(2,),
            compression="scaleoffset",
            scaleoffset=32,
            track_times=False,
        )
        expected["int32_full"] = values

        values = np.array([-300, -299, -250, -200, -199, -100], dtype=">i2")
        handle.create_dataset(
            "int16_be",
            data=values,
            chunks=(3,),
            fillvalue=-1,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        expected["int16_be"] = values

        values = np.array([-50, -7, 0, 50], dtype="i1")
        handle.create_dataset(
            "int8",
            data=values,
            chunks=(4,),
            fillvalue=-7,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        expected["int8"] = values

        values = np.array([10000, 10001, 15000, 20000], dtype=">u2")
        handle.create_dataset(
            "uint16_be",
            data=values,
            chunks=(2,),
            fillvalue=42,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        expected["uint16_be"] = values

        values = np.array([0, 1, 2, 3, 4, 5, 6], dtype="u1")
        handle.create_dataset(
            "uint8",
            data=values,
            chunks=(4,),
            fillvalue=255,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        expected["uint8"] = values

        values = np.array(
            [1000000, 1000001, 2000000, 3000000],
            dtype="<u4",
        )
        handle.create_dataset(
            "uint32",
            data=values,
            chunks=(2,),
            fillvalue=0,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        expected["uint32"] = values

        values = np.array(
            [-(2**40), -(2**40) + 1, 0, 2**40],
            dtype="<i8",
        )
        handle.create_dataset(
            "int64",
            data=values,
            chunks=(2,),
            fillvalue=-1,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        expected["int64"] = values

        values = np.array(
            [1 << 40, (1 << 40) + 5, (1 << 40) + 9, (1 << 40) + 12],
            dtype="<u8",
        )
        handle.create_dataset(
            "uint64",
            data=values,
            chunks=(2,),
            fillvalue=7,
            compression="scaleoffset",
            scaleoffset=0,
            track_times=False,
        )
        expected["uint64"] = values

        values = np.array([1.25, 2.5, -1.75, 9.5, -9.5, 0.0], dtype="<f4")
        handle.create_dataset(
            "float32_dscale",
            data=values,
            chunks=(4,),
            fillvalue=-9.5,
            compression="scaleoffset",
            scaleoffset=2,
            track_times=False,
        )
        expected["float32_dscale"] = values

        values = np.array([1.5, -2.5, 3.5, -9.5, 0.0], dtype="<f8")
        handle.create_dataset(
            "float64_dscale",
            data=values,
            chunks=(5,),
            fillvalue=-9.5,
            compression="scaleoffset",
            scaleoffset=1,
            track_times=False,
        )
        expected["float64_dscale"] = values

        values = np.array([1.25, 2.5, -1.75, 9.5, -9.5, 0.0], dtype=">f4")
        handle.create_dataset(
            "float32_be",
            data=values,
            chunks=(4,),
            fillvalue=-9.5,
            compression="scaleoffset",
            scaleoffset=2,
            track_times=False,
        )
        expected["float32_be"] = values

        values = np.array([10.25, 11.5, 12.75, -9.5, 0.0], dtype="<f4")
        handle.create_dataset(
            "float32_pipeline",
            data=values,
            chunks=(5,),
            fillvalue=-9.5,
            compression="gzip",
            compression_opts=6,
            scaleoffset=2,
            track_times=False,
        )
        expected["float32_pipeline"] = values

        values = np.full((4,), -9.5, dtype="<f4")
        handle.create_dataset(
            "float32_all_fill",
            data=values,
            chunks=(4,),
            fillvalue=-9.5,
            compression="scaleoffset",
            scaleoffset=2,
            track_times=False,
        )
        expected["float32_all_fill"] = values

    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
            filters = [
                handle[name].id.get_create_plist().get_filter(index)[0]
                for index in range(
                    handle[name].id.get_create_plist().get_nfilters()
                )
            ]
            expected_filters = [6, 1] if name == "float32_pipeline" else [6]
            assert filters == expected_filters, (name, filters)
    print(
        f"Generated {destination} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "testdata" / "scaleoffset.h5")
