#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIGS=(
  "configs/vote/full/config_fashionmnist_cnn_vote10_noise03_ns005_r101_s42.yaml"
  "configs/vote/full/config_fashionmnist_cnn_vote10_noise03_ns0001_r101_s42.yaml"
  "configs/vote/full/config_fashionmnist_cnn_vote10_noise03_ns001_r101_s42.yaml"
  "configs/vote/full/config_fashionmnist_cnn_vote10_noise03_ns0_r101_s42.yaml"
)

echo "=============================="
echo "Running Fashion-MNIST vote DP sweep sequentially"
echo "gpu=cuda:1"
echo "configs:"
printf '  - %s\n' "${CONFIGS[@]}"
echo "=============================="

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export WANDB_MODE=disabled

for config_path in "${CONFIGS[@]}"; do
  echo
  echo "[RUN] ${config_path}"
  python main.py --config "$config_path"
done

echo
echo "[SUMMARY] vote results"
python scripts/summarize_vote_results.py

echo
echo "[SUMMARY] per-client final10"
python scripts/summarize_per_client_metrics.py

echo
echo "Fashion-MNIST vote DP sweep completed."
