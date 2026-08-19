#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"
matches="$(rg -n -i 'requests\.|torch\.hub|load_state_dict_from_url|urlretrieve|wget|curl|pip install|git clone|https?://' \
  main.py inference.py ufo scripts/h200 configs/h200 || true)"
printf '%s\n' "${matches}" > offline_assets/manifests/runtime_network_scan.txt
if rg -n 'requests\.|load_state_dict_from_url|torch\.hub' main.py inference.py ufo \
  --glob '!ufo/utils/lpips.py'; then
  echo "Unexpected runtime network path found" >&2
  exit 1
fi
rg -q 'UFO_OFFLINE' ufo/utils/lpips.py
echo "RUNTIME_NETWORK_AUDIT=PASS"
