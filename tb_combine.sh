#!/usr/bin/env bash
set -euo pipefail

# Run this script from the directory that contains the 2025* experiment folders.

OUT_DIR="tensorboard"
ZIP_NAME="tensorboard.zip"

# Clean output
if [[ -d "${OUT_DIR}" ]]; then
  echo "[INFO] Removing existing ./${OUT_DIR}"
  rm -rf "${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"

# Make globs that match nothing expand to nothing (not literal patterns)
shopt -s nullglob

copied=0
skipped=0

for run_dir in 2026*/; do
  [[ -d "${run_dir}" ]] || continue

  tb_root="${run_dir%/}/tensorboard"
  if [[ ! -d "${tb_root}" ]]; then
    echo "[WARN] No tensorboard dir: ${tb_root} (skipping)"
    ((skipped++)) || true
    continue
  fi

  # Iterate exp_name directories under 2025*/tensorboard/
  for exp_dir in "${tb_root}"/*/; do
    [[ -d "${exp_dir}" ]] || continue

    exp_name="$(basename "${exp_dir%/}")"
    dest="${OUT_DIR}/${run_dir%/}__${exp_name}"
    mkdir -p "${dest}"

    tfrecords=( "${exp_dir}"/* )
    if (( ${#tfrecords[@]} == 0 )); then
      echo "[WARN] No .tfrecord files in: ${exp_dir} (skipping)"
      ((skipped++)) || true
      continue
    fi

    # Copy tfrecords
    cp -a "${tfrecords[@]}" "${dest}/"
    copied=$((copied + ${#tfrecords[@]}))
    echo "[INFO] Copied ${#tfrecords[@]} files -> ${dest}/"
  done
done

echo "[INFO] Total tfrecords copied: ${copied}"
echo "[INFO] Items skipped/warned:   ${skipped}"

# # Zip for download
# if [[ -f "${ZIP_NAME}" ]]; then
#   echo "[INFO] Removing existing ./${ZIP_NAME}"
#   rm -f "${ZIP_NAME}"
# fi

# if command -v zip >/dev/null 2>&1; then
#   echo "[INFO] Creating zip: ${ZIP_NAME}"
#   zip -r "${ZIP_NAME}" "${OUT_DIR}" >/dev/null
#   echo "[INFO] Done: ./${ZIP_NAME}"
# else
#   echo "[ERROR] 'zip' not found on this system."
#   echo "       Install zip or replace with: tar -czf tensorboard.tar.gz ${OUT_DIR}"
#   exit 1
# fi
tensorboard --logdir tensorboard