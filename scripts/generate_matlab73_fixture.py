"""Generate a MATLAB 7.3 compatible MAT-file fixture."""

from importlib.metadata import version
from pathlib import Path

import h5py
import hdf5storage
import numpy as np


def generate(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "matlab73_compatible.mat requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )
    if version("hdf5storage") != "0.2.2":
        raise SystemExit(
            "matlab73_compatible.mat requires hdf5storage 0.2.2, found "
            + version("hdf5storage")
        )

    values = {
        "temperature": np.array(
            [[21.5, 22.0], [23.25, -4.5]], dtype=np.float64
        ),
        "counts": np.arange(6, dtype=np.int32).reshape(2, 3),
        "enabled": np.array([[True, False], [False, True]], dtype=np.bool_),
        "label": "温度",
        "record": {
            "count": np.uint16(42),
            "samples": np.array([1, 2, 3], dtype=np.uint8),
        },
        "empty": np.array([], dtype=np.float64),
    }
    hdf5storage.savemat(
        destination,
        values,
        format="7.3",
        store_python_metadata=False,
        matlab_compatible=True,
    )

    with h5py.File(destination, "r") as handle:
        np.testing.assert_array_equal(
            handle["temperature"][:], values["temperature"].T
        )
        assert handle["label"].attrs["MATLAB_class"] == b"char"
        assert handle["record"].attrs["MATLAB_class"] == b"struct"
        assert handle["empty"].attrs["MATLAB_empty"] == 1
    print(
        f"Verified {destination} with h5py {h5py.__version__}, "
        f"HDF5 {h5py.version.hdf5_version}, and hdf5storage 0.2.2"
    )


if __name__ == "__main__":
    generate(
        Path(__file__).resolve().parents[1]
        / "testdata"
        / "matlab73_compatible.mat"
    )
