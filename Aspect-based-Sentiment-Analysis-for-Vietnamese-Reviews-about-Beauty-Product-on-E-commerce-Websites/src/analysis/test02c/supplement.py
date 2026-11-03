"""Aggregate TEST-02C counts without pooling seeds or altering upstream evidence."""

import csv, json, statistics
from pathlib import Path
from collections import defaultdict
from src.analysis.common import write_once
from src.analysis.test02c.pipeline import ROOT, csv_bytes

OUT = ROOT / "reports/experiments/test_02"


def read(name):
    with (OUT / name).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def stats(values):
    return (
        (statistics.mean(values), statistics.stdev(values)) if len(values) == 3 else (None, None)
    )


def main():
    groups = defaultdict(list)
    for r in read("confusion_summary.csv"):
        key = tuple(
            r[k]
            for k in ("candidate", "aspect", "definition", "gold_sentiment", "predicted_sentiment")
        )
        groups[key].append(r)
    result = []
    for key, rs in sorted(groups.items()):
        if {int(r["seed"]) for r in rs} != {42, 2026, 3407} or len(rs) != 3:
            raise ValueError("incomplete seeds")
        counts = [int(r["count"]) for r in rs]
        rates = [float(r["rate"]) for r in rs if r["rate"] != ""]
        cm, cs = stats(counts)
        rm, sd = stats(rates)
        result.append(
            dict(
                zip(
                    ("candidate", "aspect", "definition", "gold_sentiment", "predicted_sentiment"),
                    key,
                ),
                mean_count=cm,
                sd_count=cs,
                mean_rate=rm,
                sd_rate=sd,
                denominator=rs[0]["denominator"],
                seeds=3,
            )
        )
    fields = [
        "candidate",
        "aspect",
        "definition",
        "gold_sentiment",
        "predicted_sentiment",
        "mean_count",
        "sd_count",
        "mean_rate",
        "sd_rate",
        "denominator",
        "seeds",
    ]
    write_once(OUT / "confusion_seed_aggregate.csv", csv_bytes(result, fields))
    groups = defaultdict(list)
    for r in read("diagnostic_subset_comparison.csv"):
        groups[r["candidate"], r["subset"]].append(r)
    result = []
    for (candidate, subset), rs in sorted(groups.items()):
        if {int(r["seed"]) for r in rs} != {42, 2026, 3407} or len(rs) != 3:
            raise ValueError("incomplete diagnostic seeds")
        macro, macro_sd = stats([float(r["joint_macro_f1"]) for r in rs])
        micro, micro_sd = stats([float(r["joint_micro_f1"]) for r in rs])
        result.append(
            dict(
                candidate=candidate,
                subset=subset,
                reviews=rs[0]["reviews"],
                joint_macro_mean=macro,
                joint_macro_sd=macro_sd,
                joint_micro_mean=micro,
                joint_micro_sd=micro_sd,
            )
        )
    write_once(
        OUT / "diagnostic_seed_aggregate.csv",
        csv_bytes(
            result,
            [
                "candidate",
                "subset",
                "reviews",
                "joint_macro_mean",
                "joint_macro_sd",
                "joint_micro_mean",
                "joint_micro_sd",
            ],
        ),
    )
    grouped = defaultdict(lambda: defaultdict(int))
    for row in read("cross_model_error_comparison.csv"):
        correct = "CORRECT_ASPECT_CORRECT_SENTIMENT"
        tests = {
            "negative_phobert_correct_bigru_wrong": row["gold_sentiment"] == "negative"
            and row["PHOBERT_02"] == correct
            and row["BIGRU_02"] != correct,
            "negative_phobert_correct_tfidf_wrong": row["gold_sentiment"] == "negative"
            and row["PHOBERT_02"] == correct
            and row["TFIDF_02"] != correct,
            "neutral_bigru_correct_phobert_wrong": row["gold_sentiment"] == "neutral"
            and row["BIGRU_02"] == correct
            and row["PHOBERT_02"] != correct,
            "all_models_wrong": all(
                row[c] != correct for c in ("TFIDF_02", "BIGRU_02", "PHOBERT_02")
            ),
        }
        for name, match in tests.items():
            grouped[name, row["aspect"]][int(row["seed"])] += int(match)
    rows = []
    for (name, aspect), seed_counts in sorted(grouped.items()):
        counts = [seed_counts[s] for s in (42, 2026, 3407)]
        mean, sd = stats(counts)
        rows.append(
            dict(
                comparison=name,
                aspect=aspect,
                seed42=counts[0],
                seed2026=counts[1],
                seed3407=counts[2],
                mean_count=mean,
                sample_sd_count=sd,
            )
        )
    write_once(
        OUT / "cross_model_seed_aggregate.csv",
        csv_bytes(
            rows,
            [
                "comparison",
                "aspect",
                "seed42",
                "seed2026",
                "seed3407",
                "mean_count",
                "sample_sd_count",
            ],
        ),
    )
    print("Seed aggregates written")


if __name__ == "__main__":
    main()
