#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-42}"
ROUNDS="${2:-30}"
NOISE_LEVEL="${3:-0.4}"
SHIFT_COUNT=3
if [ "$#" -gt "$SHIFT_COUNT" ]; then
  shift "$SHIFT_COUNT"
  SHARE_NOISE_MULTIPLIERS=("$@")
else
  SHARE_NOISE_MULTIPLIERS=(0.1 0.3 0.5)
fi

TEMPLATE_CONFIG="configs/vote/fast/config_vote10_fast_s42.yaml"
TMP_DIR="$(mktemp -d /tmp/vote-noisy-share.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

write_config() {
  local mode="$1"
  local share_noise_multiplier="$2"
  local out_path="$TMP_DIR/config_${mode}_noise${NOISE_LEVEL}_share${share_noise_multiplier}_s${SEED}.yaml"

  python3 - "$TEMPLATE_CONFIG" "$out_path" "$SEED" "$ROUNDS" "$NOISE_LEVEL" "$mode" "$share_noise_multiplier" <<'PY'
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
seed = sys.argv[3]
rounds = sys.argv[4]
noise_level = sys.argv[5]
mode = sys.argv[6]
share_noise_multiplier = sys.argv[7]

overrides = {
    "seed": seed,
    "rounds": rounds,
    "noise_level": noise_level,
    "use_vote_pseudo": "false" if mode == "vote0" else "true",
    "vote_num_models": "0" if mode == "vote0" else "10",
    "share_noisy_vote_models": "true" if mode == "noisy_vote10" else "false",
    "share_noise_clip_norm": "1.0",
    "share_noise_multiplier": share_noise_multiplier if mode == "noisy_vote10" else "0.0",
}

lines = template_path.read_text().splitlines()
seen_keys = set()
new_lines = []
for line in lines:
    stripped = line.strip()
    replaced = False
    for key, value in overrides.items():
        prefix = f"{key}:"
        if stripped.startswith(prefix):
            new_lines.append(f"{key}: {value}")
            seen_keys.add(key)
            replaced = True
            break
    if not replaced:
        new_lines.append(line)

for key, value in overrides.items():
    if key not in seen_keys:
        new_lines.append(f"{key}: {value}")

out_path.write_text("\n".join(new_lines) + "\n")
PY

  printf '%s\n' "$out_path"
}

echo "=============================="
echo "Running noisy shared-model vote comparison"
echo "seed=${SEED}, rounds=${ROUNDS}, data_noise=${NOISE_LEVEL}"
echo "share_noise_multipliers=${SHARE_NOISE_MULTIPLIERS[*]}"
echo "runs=vote0 clean_vote10 noisy_vote10"
echo "=============================="

vote0_config="$(write_config vote0 0.0)"
echo
echo "[RUN] vote0 config=${vote0_config}"
python main.py --config "$vote0_config"

clean_vote10_config="$(write_config clean_vote10 0.0)"
echo
echo "[RUN] clean_vote10 config=${clean_vote10_config}"
python main.py --config "$clean_vote10_config"

for multiplier in "${SHARE_NOISE_MULTIPLIERS[@]}"; do
  noisy_vote10_config="$(write_config noisy_vote10 "$multiplier")"
  echo
  echo "[RUN] noisy_vote10 share_noise_multiplier=${multiplier} config=${noisy_vote10_config}"
  python main.py --config "$noisy_vote10_config"
done

echo
echo "[SUMMARY] vote results"
python scripts/summarize_vote_results.py

echo
echo "[SUMMARY] per-client final10"
python scripts/summarize_per_client_metrics.py

echo
echo "Noisy shared-model vote comparison completed."
