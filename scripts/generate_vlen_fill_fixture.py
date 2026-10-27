from pathlib import Path

import h5py


def require_hdf5_1_14_6() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            "variable-length fill fixture requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )


def generate(destination: Path) -> None:
    require_hdf5_1_14_6()
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(destination, "w", libver="latest") as handle:
        handle.create_dataset(
            "contiguous",
            shape=(4,),
            dtype=string_type,
            fillvalue="missing",
            track_times=False,
        )

        chunked = handle.create_dataset(
            "chunked",
            shape=(6,),
            dtype=string_type,
            chunks=(2,),
            fillvalue="fallback",
            track_times=False,
        )
        chunked[0] = "first"
        chunked[5] = "last"

        extensible = handle.create_dataset(
            "extensible",
            shape=(0,),
            maxshape=(None,),
            dtype=string_type,
            chunks=(2,),
            fillvalue="grow",
            track_times=False,
        )
        extensible.resize((5,))


if __name__ == "__main__":
    generate(Path("testdata/vlen_fills.h5"))
