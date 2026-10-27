"""Verify the committed HDF5 1.14.6 object-reference fixture."""

from pathlib import Path
import sys

import h5py
import numpy as np


def verify(path: Path) -> None:
    expected = {
        0: ("/target", "Dataset"),
        1: ("/group", "Group"),
        2: ("/group/child", "Dataset"),
        3: ("/", "Group"),
        4: ("/named_i4", "Datatype"),
    }
    with h5py.File(path, "r", libver="latest") as handle:
        references = handle["object_refs"][:]
        assert handle["object_refs"].id.get_type().get_size() == 8
        assert handle["region_refs"].id.get_type().get_size() == 12
        for index, (expected_path, expected_kind) in expected.items():
            resolved = handle[references[index]]
            assert resolved.name == expected_path
            assert type(resolved).__name__ == expected_kind
        assert not bool(references[5])

        child_reference = handle["target"].attrs["child_ref"]
        assert handle[child_reference[0]].name == "/group/child"

        dangling = handle["dangling_refs"][0]
        try:
            handle[dangling]
        except KeyError:
            pass
        else:
            raise AssertionError("dangling object reference unexpectedly resolved")

        region = handle["region_refs"][0]
        np.testing.assert_array_equal(handle[region][region], [1, 2])

    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_reference_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
