# fedpll

## Directory Layout

- `main.py`, `client.py`, `server.py`, `common.py`, `model.py`: core training code
- `config.yaml`: current default config
- `configs/legacy/`: older baseline and comparison configs
- `configs/vote/full/`: full-length vote ablation configs
- `configs/vote/fast/`: shorter vote configs for quick screening, including fixed-seed variants
- `analysis/`: prototype verification and other analysis scripts
- `notebooks/`: local exploratory notebooks
- `local/`: local-only files not needed for project execution
- `scripts/summarize_vote_results.py`: aggregate `round_metrics.csv` files across runs
- `csv_logs/`: per-run exported metrics and q-vector logs
- `logs/`: runtime log files
- `data/`: downloaded datasets
- `temp/`: local scratch files and moved historical artifacts

## Common Commands

Run a fast vote experiment:

```bash
python main.py --config configs/vote/fast/config_vote3_fast_s42.yaml
```

Summarize completed vote runs:

```bash
python scripts/summarize_vote_results.py
```
