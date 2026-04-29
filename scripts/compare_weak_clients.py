import argparse
import csv
import glob
import os
import re
from statistics import mean


COMPARE_FIELDS = [
    "client_disamb_q_acc",
    "client_disamb_q_macro_recall",
    "client_disamb_q_macro_f1",
    "client_personalized_test_acc",
    "client_personalized_test_macro_f1",
]


def infer_vote_tag(run_dir: str) -> str:
    lower_name = os.path.basename(run_dir).lower()
    for tag in ("vote10", "vote5", "vote3", "vote1", "novote", "vote0"):
        if tag in lower_name:
            return "vote0" if tag == "novote" else tag
    return "unknown"


def infer_noise_level(run_dir: str) -> str:
    lower_name = os.path.basename(run_dir).lower()
    match = re.search(r"noise([0-9]+(?:\.[0-9]+)?)", lower_name)
    if match:
        return match.group(1)
    return "unknown"


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_rows(path: str):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def summarize_final_window(run_dir: str, final_rounds: int):
    csv_path = os.path.join(run_dir, "per_client_round_metrics.csv")
    if not os.path.exists(csv_path):
        return []

    rows = load_rows(csv_path)
    if not rows:
        return []

    max_round = max(int(float(row["round"])) for row in rows)
    min_round = max_round - final_rounds + 1
    rows_by_client = {}
    for row in rows:
        round_id = int(float(row["round"]))
        if round_id < min_round:
            continue
        client_id = int(float(row["client_id"]))
        rows_by_client.setdefault(client_id, []).append(row)

    out_rows = []
    for client_id, client_rows in sorted(rows_by_client.items()):
        out = {
            "run_dir": os.path.basename(run_dir),
            "noise_level": infer_noise_level(run_dir),
            "vote_tag": infer_vote_tag(run_dir),
            "client_id": client_id,
            "final_rounds": len(client_rows),
            "first_round": min(int(float(row["round"])) for row in client_rows),
            "last_round": max(int(float(row["round"])) for row in client_rows),
        }
        for field in COMPARE_FIELDS:
            values = [to_float(row.get(field)) for row in client_rows]
            out[field] = mean(values) if values else 0.0
        out_rows.append(out)
    return out_rows


def build_summary_rows(run_dirs, final_rounds: int):
    rows = []
    for run_dir in sorted(run_dirs):
        rows.extend(summarize_final_window(run_dir, final_rounds))
    return rows


def build_comparison_rows(summary_rows, weak_top_k: int):
    grouped = {}
    for row in summary_rows:
        key = (row["noise_level"], row["client_id"])
        grouped.setdefault(key, {})[row["vote_tag"]] = row

    comparison_rows = []
    for (noise_level, client_id), vote_rows in sorted(grouped.items()):
        vote0 = vote_rows.get("vote0")
        if vote0 is None:
            continue
        out = {
            "noise_level": noise_level,
            "client_id": client_id,
            "vote0_client_personalized_test_acc": vote0["client_personalized_test_acc"],
            "vote0_client_personalized_test_macro_f1": vote0["client_personalized_test_macro_f1"],
            "vote0_client_disamb_q_acc": vote0["client_disamb_q_acc"],
            "vote0_client_disamb_q_macro_f1": vote0["client_disamb_q_macro_f1"],
        }
        for vote_tag in ("vote5", "vote10"):
            vote_row = vote_rows.get(vote_tag)
            if vote_row is None:
                for field in COMPARE_FIELDS:
                    out[f"{vote_tag}_{field}"] = ""
                    out[f"{vote_tag}_minus_vote0_{field}"] = ""
                continue
            for field in COMPARE_FIELDS:
                out[f"{vote_tag}_{field}"] = vote_row[field]
                out[f"{vote_tag}_minus_vote0_{field}"] = vote_row[field] - vote0[field]
        comparison_rows.append(out)

    comparison_rows.sort(key=lambda row: (row["noise_level"], row["vote0_client_personalized_test_acc"]))

    if weak_top_k > 0:
        filtered = []
        by_noise = {}
        for row in comparison_rows:
            by_noise.setdefault(row["noise_level"], []).append(row)
        for noise_level, rows in sorted(by_noise.items()):
            filtered.extend(rows[:weak_top_k])
        return filtered
    return comparison_rows


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
    parser = argparse.ArgumentParser(description="Compare weak-client changes across vote settings at the same noise level.")
    parser.add_argument("--logs-dir", type=str, default="csv_logs")
    parser.add_argument("--pattern", type=str, default="*vote*")
    parser.add_argument("--final-rounds", type=int, default=10)
    parser.add_argument("--weak-top-k", type=int, default=3, help="Keep the weakest K clients per noise based on vote0 personalized accuracy. Use 0 to keep all.")
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

    summary_rows = build_summary_rows(run_dirs, args.final_rounds)
    comparison_rows = build_comparison_rows(summary_rows, args.weak_top_k)

    fieldnames = [
        "noise_level",
        "client_id",
        "vote0_client_personalized_test_acc",
        "vote0_client_personalized_test_macro_f1",
        "vote0_client_disamb_q_acc",
        "vote0_client_disamb_q_macro_f1",
    ]
    for vote_tag in ("vote5", "vote10"):
        for field in COMPARE_FIELDS:
            fieldnames.append(f"{vote_tag}_{field}")
            fieldnames.append(f"{vote_tag}_minus_vote0_{field}")

    os.makedirs(args.logs_dir, exist_ok=True)
    out_path = os.path.join(args.logs_dir, f"weak_client_comparison_final{args.final_rounds}.csv")
    write_csv(out_path, comparison_rows, fieldnames)

    print_table(
        "Weak-client comparison",
        comparison_rows,
        [
            "noise_level",
            "client_id",
            "vote0_client_personalized_test_acc",
            "vote5_minus_vote0_client_personalized_test_acc",
            "vote10_minus_vote0_client_personalized_test_acc",
            "vote5_minus_vote0_client_disamb_q_macro_f1",
            "vote10_minus_vote0_client_disamb_q_macro_f1",
        ],
    )
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
