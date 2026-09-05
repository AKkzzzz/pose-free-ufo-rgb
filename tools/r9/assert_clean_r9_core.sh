#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE=10445abf3f1dcfba8f5401c1a94b06c6c310c132
CORE=(
  ufo/models/archs/small.py
  ufo/models/sam_object_detail_r9.py
  ufo/models/sam_object_motion_r5.py
  ufo/models/sam_object_motion_r6.py
  ufo/dataset/dataset.py
  ufo/dataset/data_utils.py
  ufo/dataset/sam_tracks.py
  inference.py
)
cd "${ROOT}"
diff_output="$(git diff "${BASE}" -- "${CORE[@]}")"
if [[ -n "${diff_output}" ]]; then
  printf '%s\n' "${diff_output}" >&2
  echo "CLEAN_R9_CORE=NOT_EMPTY" >&2
  exit 1
fi
echo "CLEAN_R9_CORE=EMPTY"
