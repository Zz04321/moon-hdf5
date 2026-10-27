#!/bin/sh

set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

echo "==> Update interfaces"
moon info

echo "==> Check formatting"
moon fmt --check

echo "==> Check"
moon check

echo "==> Test (wasm)"
moon test --target wasm

echo "==> Test (native)"
moon test --target native

echo "==> Documentation"
moon doc

echo "==> Package contents"
moon package --list >/dev/null 2>&1 || {
  moon package --list
  exit 1
}

echo "==> Writer compatibility"
output=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-release.XXXXXX")
trap 'rm -f "$output"' EXIT HUP INT TERM
moon run src/cmd/write_demo --target native -- "$output"

python=${H5PY_PYTHON:-}
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> h5py verification skipped (SKIP_H5PY=1)"
else
  if [ -z "$python" ]; then
    for candidate in python3 python; do
      if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c "import h5py, numpy" >/dev/null 2>&1; then
        python=$candidate
        break
      fi
    done
  fi
  if [ -z "$python" ]; then
    echo "h5py and NumPy are required; set H5PY_PYTHON or SKIP_H5PY=1" >&2
    exit 1
  fi
  "$python" scripts/verify_writer_output.py "$output"
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Numeric layout fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/numeric_layouts.h5 ]; then
  echo "==> Numeric layout fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_numeric_layouts_fixture.py testdata/numeric_layouts.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> NetCDF-4 fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/netcdf4_1_7_5.h5 ]; then
  echo "==> NetCDF-4 fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_netcdf4_fixture.py testdata/netcdf4_1_7_5.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> External raw-data fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/external_storage.h5 ]; then
  echo "==> External raw-data fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_external_storage_fixture.py testdata/external_storage.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Shared-message fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/shared_messages.h5 ]; then
  echo "==> Shared-message fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_shared_messages_fixture.py testdata/shared_messages.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Reference fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/references.h5 ] ||
  [ ! -f testdata/references_v108.h5 ] ||
  [ ! -f testdata/references_layouts_32.h5 ]; then
  echo "==> Reference fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_reference_fixture.py testdata/references.h5
  "$python" scripts/verify_reference_fixture.py testdata/references_v108.h5
  "$python" scripts/verify_region_layout_fixture.py \
    testdata/references_layouts_32.h5
fi

echo "Release checks passed."
