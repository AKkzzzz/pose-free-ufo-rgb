#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"

mkdir -p offline_assets/manifests

PATTERN='requests\.|torch\.hub|load_state_dict_from_url|urlretrieve|wget|curl|pip install|git clone|https?://'

if command -v rg >/dev/null 2>&1; then
    # Full scan for record
    rg -n -i "${PATTERN}" \
        main.py inference.py ufo scripts/h200 configs/h200 \
        > offline_assets/manifests/runtime_network_scan.txt || true

    # Runtime-critical scan, excluding the intentionally offline-patched LPIPS file
    if rg -n 'requests\.|load_state_dict_from_url|torch\.hub' \
        main.py inference.py ufo \
        --glob '!ufo/utils/lpips.py'; then
        echo "Unexpected runtime network path found" >&2
        exit 1
    fi

    rg -q 'UFO_OFFLINE' ufo/utils/lpips.py

else
    echo "[INFO] ripgrep not found, falling back to grep"

    # Full scan for record
    grep -RInE "${PATTERN}" \
        main.py inference.py ufo scripts/h200 configs/h200 \
        > offline_assets/manifests/runtime_network_scan.txt || true

    # Runtime-critical scan; exclude lpips.py
    if grep -RInE \
        'requests\.|load_state_dict_from_url|torch\.hub' \
        main.py inference.py ufo \
        --exclude='lpips.py'; then
        echo "Unexpected runtime network path found" >&2
        exit 1
    fi

    grep -q 'UFO_OFFLINE' ufo/utils/lpips.py
fi

echo "RUNTIME_NETWORK_AUDIT=PASS"