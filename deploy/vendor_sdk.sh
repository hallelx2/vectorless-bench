#!/usr/bin/env bash
# Stage the monorepo's vectorless SDK into the Docker build context.
# The SDK isn't published to PyPI, so the image installs it from ./vendor/.
# Run this before `docker compose build` / shipping the bundle to a VM.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"          # vectorless-bench/
SDK_SRC="${SDK_SRC:-$HERE/../vectorless-sdk/python}"  # sibling in the monorepo
DEST="$HERE/vendor/vectorless-sdk"

if [ ! -f "$SDK_SRC/pyproject.toml" ] && [ ! -f "$SDK_SRC/setup.py" ]; then
  echo "SDK source not found at $SDK_SRC (set SDK_SRC=...)" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
# copy source only — skip local build/venv cruft
cp -r "$SDK_SRC" "$DEST"
rm -rf "$DEST/.venv" "$DEST/dist" "$DEST/build" "$DEST"/*.egg-info 2>/dev/null || true
echo "vendored SDK: $SDK_SRC -> $DEST"
