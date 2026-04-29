#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-42}"
ROUNDS="${2:-30}"
NOISE_LEVELS=("${@:3}")
if [ "${#NOISE_LEVELS[@]}" -eq 0 ]; then
  NOISE_LEVELS=(0.3 0.4 0.5 0.6)
fi

TEMPLATE_CONFIG="configs/vote/fast/config_vote0_fast_s42.yaml"
TMP_DIR="$(mktemp -d /tmp/vote-noise-sweep.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

write_config() {
  local noise="$1"
  local vote="$2"
  local use_vote="false"

  if [ "$vote" -gt 0 ]; then
    use_vote="true"
  fi

  local out_path="$TMP_DIR/config_noise${noise}_vote${vote}_s${SEED}.yaml"
  python3 - "$TEMPLATE_CONFIG" "$out_path" "$SEED" "$ROUNDS" "$noise" "$use_vote" "$vote" <<'PY'
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
seed = sys.argv[3]
rounds = sys.argv[4]
noise = sys.argv[5]
use_vote = sys.argv[6]
vote_num = sys.argv[7]

overrides = {
    "seed": seed,
    "rounds": rounds,
    "noise_level": noise,
    "use_vote_pseudo": use_vote,
    "vote_num_models": vote_num,
}

lines = template_path.read_text().splitlines()
new_lines = []
for line in lines:
    stripped = line.strip()
    replaced = False
    for key, value in overrides.items():
        prefix = f"{key}:"
        if stripped.startswith(prefix):
            new_lines.append(f"{key}: {value}")
            replaced = True
            break
    if not replaced:
        new_lines.append(line)

out_path.write_text("\n".join(new_lines) + "\n")
PY
  printf '%s\n' "$out_path"
}

echo "=============================="
echo "Running single-seed noise sweep"
echo "seed=${SEED}, rounds=${ROUNDS}"
echo "noise_levels=${NOISE_LEVELS[*]}"
echo "votes=0 5 10"
echo "=============================="

for noise in "${NOISE_LEVELS[@]}"; do
  echo
  echo "---------- noise=${noise} ----------"
  for vote in 0 5 10; do
    config_path="$(write_config "$noise" "$vote")"
    echo "[RUN] noise=${noise} vote=${vote} config=${config_path}"
    python main.py --config "$config_path"
  done
done

echo
echo "[SUMMARY] vote results"
python scripts/summarize_vote_results.py

echo
echo "[SUMMARY] per-client final10"
python scripts/summarize_per_client_metrics.py

echo
echo "Single-seed noise sweep completed."
