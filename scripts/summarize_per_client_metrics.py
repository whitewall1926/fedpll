import argparse
import csv
import glob
import os
from statistics import mean, pstdev


PER_CLIENT_FIELDS = [
    "client_disamb_q_acc",
    "client_disamb_q_macro_recall",
    "client_disamb_q_macro_precision",
    "client_disamb_q_macro_f1",
    "client_personalized_test_acc",
    "client_personalized_test_macro_recall",
    "client_personalized_test_macro_precision",
    "client_personalized_test_macro_f1",
]


def infer_vote_tag(run_dir: str) -> str:
    lower_name = os.path.basename(run_dir).lower()
    for tag in ("vote10", "vote5", "vote3", "vote1", "novote", "vote0"):
        if tag in lower_name:
            return "vote0" if tag == "novote" else tag
    return "unknown"


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_rows(path: str):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def build_final_window_rows(run_dirs, final_rounds: int):
    summary_rows = []
    for run_dir in sorted(run_dirs):
        csv_path = os.path.join(run_dir, "per_client_round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        rows = load_rows(csv_path)
        if not rows:
            continue

        max_round = max(int(float(row["round"])) for row in rows)
        min_round = max_round - final_rounds + 1
        rows_by_client = {}
        for row in rows:
            round_id = int(float(row["round"]))
            if round_id < min_round:
                continue
            client_id = int(float(row["client_id"]))
            rows_by_client.setdefault(client_id, []).append(row)

        for client_id, client_rows in sorted(rows_by_client.items()):
            out = {
                "run_dir": os.path.basename(run_dir),
                "vote_tag": infer_vote_tag(run_dir),
                "client_id": client_id,
                "final_rounds": len(client_rows),
                "first_round": min(int(float(row["round"])) for row in client_rows),
                "last_round": max(int(float(row["round"])) for row in client_rows),
            }
            for field in PER_CLIENT_FIELDS:
                values = [to_float(row.get(field)) for row in client_rows]
                out[f"{field}_mean"] = mean(values)
                out[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
            summary_rows.append(out)
    return summary_rows


def aggregate_by_vote_and_client(summary_rows):
    grouped = {}
    for row in summary_rows:
        key = (row["vote_tag"], row["client_id"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (vote_tag, client_id), rows in sorted(grouped.items()):
        out = {
            "vote_tag": vote_tag,
            "client_id": client_id,
            "num_runs": len(rows),
        }
        for field in PER_CLIENT_FIELDS:
            values = [row[f"{field}_mean"] for row in rows]
            out[f"{field}_mean"] = mean(values)
            out[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(out)
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

    widths = {
        field: max(len(field), max(len(f"{row.get(field, '')}") for row in rows))
        for field in fields
    }
    print("  " + " | ".join(field.ljust(widths[field]) for field in fields))
    print("  " + "-+-".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  " + " | ".join(f"{row.get(field, '')}".ljust(widths[field]) for field in fields))


def main():
    parser = argparse.ArgumentParser(description="Summarize final-window per-client metrics.")
    parser.add_argument("--logs-dir", type=str, default="csv_logs")
    parser.add_argument("--pattern", type=str, default="*vote*")
    parser.add_argument("--final-rounds", type=int, default=10)
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

    summary_rows = build_final_window_rows(run_dirs, args.final_rounds)
    aggregate_rows = aggregate_by_vote_and_client(summary_rows)

    run_fields = [
        "run_dir",
        "vote_tag",
        "client_id",
        "final_rounds",
        "first_round",
        "last_round",
    ] + [f"{field}_{suffix}" for field in PER_CLIENT_FIELDS for suffix in ("mean", "std")]
    group_fields = [
        "vote_tag",
        "client_id",
        "num_runs",
    ] + [f"{field}_{suffix}" for field in PER_CLIENT_FIELDS for suffix in ("mean", "std")]

    os.makedirs(args.logs_dir, exist_ok=True)
    window_label = f"final{args.final_rounds}"
    write_csv(os.path.join(args.logs_dir, f"per_client_{window_label}_run_summary.csv"), summary_rows, run_fields)
    write_csv(os.path.join(args.logs_dir, f"per_client_{window_label}_group_summary.csv"), aggregate_rows, group_fields)

    print_table(
        "Per-client final-window personalized test accuracy",
        aggregate_rows,
        ["vote_tag", "client_id", "num_runs", "client_personalized_test_acc_mean", "client_personalized_test_acc_std"],
    )


if __name__ == "__main__":
    main()
