#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_group() {
  local seed="$1"
  shift

  echo "=============================="
  echo "Running fast vote experiments for seed=${seed}"
  echo "=============================="

  for config in "$@"; do
    echo
    echo "[RUN] $config"
    python main.py --config "$config"
  done

  echo
  echo "[SUMMARY] seed=${seed}"
  python scripts/summarize_vote_results.py
}

run_group 42 \
  configs/vote/fast/config_vote0_fast_s42.yaml \
  configs/vote/fast/config_vote1_fast_s42.yaml \
  configs/vote/fast/config_vote3_fast_s42.yaml \
  configs/vote/fast/config_vote5_fast_s42.yaml \
  configs/vote/fast/config_vote10_fast_s42.yaml

run_group 123 \
  configs/vote/fast/config_vote0_fast_s123.yaml \
  configs/vote/fast/config_vote1_fast_s123.yaml \
  configs/vote/fast/config_vote3_fast_s123.yaml \
  configs/vote/fast/config_vote5_fast_s123.yaml \
  configs/vote/fast/config_vote10_fast_s123.yaml

run_group 3407 \
  configs/vote/fast/config_vote0_fast_s3407.yaml \
  configs/vote/fast/config_vote1_fast_s3407.yaml \
  configs/vote/fast/config_vote3_fast_s3407.yaml \
  configs/vote/fast/config_vote5_fast_s3407.yaml \
  configs/vote/fast/config_vote10_fast_s3407.yaml

echo
echo "All fast vote experiments completed."
