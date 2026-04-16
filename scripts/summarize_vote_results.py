import argparse
import csv
import glob
import os
from statistics import mean, pstdev


METRIC_FIELDS = [
    "server_test_acc",
    "server_test_balanced_acc",
    "server_test_macro_f1",
    "global_disamb_acc",
    "global_disamb_balanced_acc",
    "global_disamb_macro_f1",
    "client_disamb_std",
    "client_test_acc_mean",
    "client_test_acc_std",
    "client_vote_pseudo_acc_mean",
    "client_vote_confidence_mean",
]


def infer_vote_tag(run_dir: str) -> str:
    lower_name = os.path.basename(run_dir).lower()
    for tag in ("vote10", "vote5", "vote3", "vote1", "novote", "vote0"):
        if tag in lower_name:
            return "vote0" if tag == "novote" else tag
    return "unknown"


def load_last_row(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_summary_rows(run_dirs):
    summary_rows = []
    for run_dir in sorted(run_dirs):
        csv_path = os.path.join(run_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        last_row = load_last_row(csv_path)
        if last_row is None:
            continue

        row = {
            "run_dir": os.path.basename(run_dir),
            "vote_tag": infer_vote_tag(run_dir),
            "last_round": int(float(last_row["round"])),
        }
        for field in METRIC_FIELDS:
            row[field] = to_float(last_row.get(field))
        summary_rows.append(row)
    return summary_rows


def aggregate_by_vote_tag(summary_rows):
    grouped = {}
    for row in summary_rows:
        grouped.setdefault(row["vote_tag"], []).append(row)

    aggregate_rows = []
    for vote_tag, rows in sorted(grouped.items()):
        agg = {
            "vote_tag": vote_tag,
            "num_runs": len(rows),
        }
        for field in METRIC_FIELDS:
            values = [r[field] for r in rows]
            agg[f"{field}_mean"] = mean(values)
            agg[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(agg)
    return aggregate_rows


def write_csv(path: str, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, rows, fields):
    print(title)
    if not rows:
        print("  no rows found")
        return

    widths = {}
    for field in fields:
        widths[field] = max(len(field), max(len(f"{row.get(field, '')}") for row in rows))

    header = "  " + " | ".join(field.ljust(widths[field]) for field in fields)
    print(header)
    print("  " + "-+-".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  " + " | ".join(f"{row.get(field, '')}".ljust(widths[field]) for field in fields))


def main():
    parser = argparse.ArgumentParser(description="Summarize vote experiment CSV outputs.")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="csv_logs",
        help="Directory containing per-run folders with round_metrics.csv",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*vote*",
        help="Glob pattern under logs-dir used to select run folders",
    )
    args = parser.parse_args()

    run_dirs = [
        path for path in glob.glob(os.path.join(args.logs_dir, args.pattern))
        if os.path.isdir(path)
    ]
    if not run_dirs:
        run_dirs = [
            path for path in glob.glob(os.path.join(args.logs_dir, "*"))
            if os.path.isdir(path)
        ]

    summary_rows = build_summary_rows(run_dirs)
    aggregate_rows = aggregate_by_vote_tag(summary_rows)

    os.makedirs(args.logs_dir, exist_ok=True)
    write_csv(
        os.path.join(args.logs_dir, "vote_run_summary.csv"),
        summary_rows,
        ["run_dir", "vote_tag", "last_round"] + METRIC_FIELDS,
    )
    write_csv(
        os.path.join(args.logs_dir, "vote_group_summary.csv"),
        aggregate_rows,
        ["vote_tag", "num_runs"] + [f"{field}_{suffix}" for field in METRIC_FIELDS for suffix in ("mean", "std")],
    )

    print_table(
        "Per-run summary",
        summary_rows,
        ["run_dir", "vote_tag", "server_test_acc", "global_disamb_balanced_acc", "client_vote_pseudo_acc_mean"],
    )
    print()
    print_table(
        "Grouped summary",
        aggregate_rows,
        ["vote_tag", "num_runs", "server_test_acc_mean", "global_disamb_balanced_acc_mean", "client_vote_pseudo_acc_mean_mean"],
    )


if __name__ == "__main__":
    main()
