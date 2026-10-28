import sys
from pathlib import Path

import h5py
import numpy as np


def verify(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        assert list(handle) == ["a", "b"]
        assert list(handle["a"]) == ["loop", "values"]
        a_address = h5py.h5o.get_info(handle["a"].id).addr
        assert h5py.h5o.get_info(handle["b"].id).addr == a_address
        assert h5py.h5o.get_info(handle["a/loop"].id).addr == a_address
        for name in ("a/values", "b/values", "a/loop/values", "b/loop/values"):
            np.testing.assert_array_equal(handle[name][:], [1, 2, 3])
    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_hard_links_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
