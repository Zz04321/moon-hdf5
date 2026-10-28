"""Generate the HDF5 1.14.6 hard-link alias and cycle fixture."""

from pathlib import Path

import h5py
import numpy as np


def generate(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "hard_links.h5 requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    with h5py.File(destination, "w", libver="latest") as handle:
        group = handle.create_group("a")
        group.create_dataset(
            "values",
            data=np.array([1, 2, 3], dtype="<i4"),
            track_times=False,
        )
        handle["b"] = group
        group["loop"] = group

    with h5py.File(destination, "r") as handle:
        assert list(handle["a"]) == ["loop", "values"]
        np.testing.assert_array_equal(handle["b/values"][:], [1, 2, 3])
        assert (
            h5py.h5o.get_info(handle["a/loop"].id).addr
            == h5py.h5o.get_info(handle["a"].id).addr
        )
    print(
        f"Verified {destination} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "testdata" / "hard_links.h5")
