from pathlib import Path
import sys

import h5py
import numpy as np


def integer_layout(datatype) -> tuple[int, int, bool, str]:
    assert isinstance(datatype, h5py.h5t.TypeIntegerID)
    assert datatype.get_size() == 4
    order = "little" if datatype.get_order() == h5py.h5t.ORDER_LE else "big"
    return (
        datatype.get_precision(),
        datatype.get_offset(),
        datatype.get_sign() == h5py.h5t.SGN_2,
        order,
    )


def verify(path: Path) -> None:
    cases = {
        "offset": (16, 8, True, "little", [42, -2, -32768]),
        "precision": (16, 0, True, "little", [42, -2, -32768]),
        "unsigned": (20, 4, False, "little", [0, 1, 703710, 1048575]),
        "big_endian": (20, 4, True, "big", [-2, 42, 524287]),
    }
    with h5py.File(path, "r") as handle:
        for name, (precision, offset, signed, order, values) in cases.items():
            dataset = handle[name]
            assert integer_layout(dataset.id.get_type()) == (
                precision,
                offset,
                signed,
                order,
            )
            np.testing.assert_array_equal(dataset[:], values)
            attribute = dataset.attrs.get_id("custom")
            assert integer_layout(attribute.get_type()) == (
                precision,
                offset,
                signed,
                order,
            )
            np.testing.assert_array_equal(dataset.attrs["custom"], values)

        fill = handle["fill"]
        assert integer_layout(fill.id.get_type()) == (16, 8, True, "little")
        assert fill.fillvalue == -2
        np.testing.assert_array_equal(fill[:], [-2, -2, -2])

        custom_float = handle["custom_float"].id.get_type()
        assert isinstance(custom_float, h5py.h5t.TypeFloatID)
        assert custom_float.get_ebias() == 126
        np.testing.assert_array_equal(handle["custom_float"][:], [1.5])

        for endian in ("<", ">"):
            prefix = "le" if endian == "<" else "be"
            for kind in ("i", "u", "f"):
                for width in ((2, 4, 8) if kind == "f" else (1, 2, 4, 8)):
                    name = f"{prefix}_{kind}{width}"
                    np.testing.assert_array_equal(handle[name][:], [1, 42])
        half = handle["half_special"]
        assert half.dtype == np.dtype("<f2")
        np.testing.assert_array_equal(
            half[:],
            np.array(
                [
                    1.0,
                    -2.5,
                    65504.0,
                    6.103515625e-05,
                    5.960464477539063e-08,
                    0.0,
                    -0.0,
                ],
                dtype="<f2",
            ),
        )
        assert not np.signbit(half[5])
        assert np.signbit(half[6])
        np.testing.assert_array_equal(
            half.attrs["custom"], [1.0, -2.5, 65504.0]
        )
        np.testing.assert_array_equal(handle["half_fill"][:], [-2.5, -2.5, -2.5])

    print(
        f"Verified {path.name} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_numeric_layouts_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
