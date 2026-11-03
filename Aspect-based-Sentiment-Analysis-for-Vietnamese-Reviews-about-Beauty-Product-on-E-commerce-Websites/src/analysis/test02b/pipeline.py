"""Quantitative analysis of sealed predictions; independent of the CLI."""

import json
import math
import statistics
import sys

from src.analysis.common import ROOT, csv_bytes, digest, write_once
from .core import (
    CANDIDATES,
    SEEDS,
    SENTIMENTS,
    decompose_contributions,
    validate_pair_counts,
    validate_seed_matrix,
    verify_decomposition,
)

# The frozen evaluator uses its original top-level imports.
sys.path.insert(0, str(ROOT / "src"))
from evaluate_predictions import evaluate_file


def save(path, rows):
    """Compatibility helper: serialize rows without writing the supplied path."""
    return csv_bytes(rows)


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if type(a) in (float, int) and type(b) in (float, int):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    return a == b


def read_verified_runs(runs):
    """Load all nine runs and reconcile predictions, metrics and summary."""
    summary = json.loads((runs / "final_test_summary.json").read_text())
    results = {}
    sources = {}
    for c in CANDIDATES:
        for seed in SEEDS:
            rid = f"{c}-seed-{seed}"
            d = runs / rid / "final_test"
            pred = d / "test_predictions.jsonl"
            metric = d / "test_metrics.json"
            manifest = json.loads((d / "inference_manifest.json").read_text())
            stored = json.loads(metric.read_text())
            if (
                digest(pred) != manifest["prediction_sha256"]
                or digest(pred) != (d / "test_predictions.sha256").read_text().strip()
                or digest(metric) != manifest["metrics_sha256"]
            ):
                raise ValueError(f"hash mismatch {rid}")
            calculated = evaluate_file(pred, "test", ROOT)
            if not close(calculated, stored):
                raise ValueError(f"frozen evaluator mismatch {rid}")
            saved = next(x for x in summary["families"][c]["per_seed"] if x["seed"] == seed)
            if not close(saved["reports"], stored["reports"]):
                raise ValueError(f"summary mismatch {rid}")
            results[c, seed] = stored["reports"]["original_test"]["joint"]
            sources[rid] = {"prediction": digest(pred), "metrics": digest(metric)}
            print(rid, "verified", flush=True)
    validate_seed_matrix(results.keys())
    labels = list(results[CANDIDATES[0], 42]["per_label"])
    if len(labels) != 21 or any(
        set(results[c, s]["per_label"]) != set(labels) for c in CANDIDATES for s in SEEDS
    ):
        raise ValueError("fixed 21 labels mismatch")
    return results, sources, labels


def pairwise_tables(results, labels):
    """Keep per-seed counts and paired differences for every label."""
    pair = []
    deltas = []
    for label in labels:
        for c in CANDIDATES:
            vals = [results[c, s]["per_label"][label]["f1"] for s in SEEDS]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals)
            for s in SEEDS:
                m = results[c, s]["per_label"][label]
                validate_pair_counts(m)
                pair.append(
                    dict(
                        label=label,
                        candidate=c,
                        seed=s,
                        **{
                            k: m[k]
                            for k in ("support", "tp", "fp", "fn", "precision", "recall", "f1")
                        },
                        mean_f1=mean,
                        sample_sd_f1=sd,
                    )
                )
        for a, b in [
            ("PHOBERT-02", "BIGRU-02"),
            ("PHOBERT-02", "TFIDF-02"),
            ("BIGRU-02", "TFIDF-02"),
        ]:
            diff = [
                results[a, s]["per_label"][label]["f1"] - results[b, s]["per_label"][label]["f1"]
                for s in SEEDS
            ]
            deltas.append(
                dict(
                    label=label,
                    comparison=f"{a} minus {b}",
                    support=results[a, 42]["per_label"][label]["support"],
                    mean_f1_difference=statistics.mean(diff),
                    sample_sd=statistics.stdev(diff),
                )
            )
    return pair, deltas


def sentiment_contributions(results, labels):
    """Decompose each paired Macro-F1 difference without pooling seed counts."""
    decomposition = []
    for a, b in [("PHOBERT-02", "BIGRU-02"), ("PHOBERT-02", "TFIDF-02"), ("BIGRU-02", "TFIDF-02")]:
        for s in SEEDS:
            rows = decompose_contributions(
                results[a, s]["per_label"], results[b, s]["per_label"], labels, f"{a} minus {b}", s
            )
            verify_decomposition(rows, results[a, s]["macro_f1"], results[b, s]["macro_f1"])
            decomposition.extend(rows)
    return decomposition


def run_analysis():
    runs = ROOT / "absa-runs-final-test"
    out = ROOT / "reports/experiments/test_02"
    results, sources, labels = read_verified_runs(runs)
    pair, deltas = pairwise_tables(results, labels)
    decomposition = sentiment_contributions(results, labels)
    payloads = {
        name: save(out / name, items)
        for name, items in [
            ("per_pair_metrics.csv", pair),
            ("per_pair_deltas.csv", deltas),
            ("macro_f1_decomposition.csv", decomposition),
        ]
    }
    text = [
        "# TEST-02B quantitative analysis",
        "",
        "All nine predictions were re-evaluated with the unchanged frozen evaluator. Every report matched its stored TEST-01 report and summary (relative tolerance 1e-10, absolute tolerance 1e-12).",
        "",
        "The pair table reports all 21 labels with seed-level counts and sample SD; the decomposition uses sum of seven pair differences divided by 21 and reconciles for every seed. Pooled sentiment F1 is a different quantity.",
        "",
        "| Comparison | Positive contribution | Neutral contribution | Negative contribution |",
        "|---|---:|---:|---:|",
    ]
    for a, b in [("PHOBERT-02", "BIGRU-02"), ("PHOBERT-02", "TFIDF-02"), ("BIGRU-02", "TFIDF-02")]:
        comp = f"{a} minus {b}"
        vals = [
            statistics.mean(
                x["fixed21_contribution"]
                for x in decomposition
                if x["comparison"] == comp and x["sentiment"] == sent
            )
            for sent in SENTIMENTS
        ]
        text.append("| " + comp + " | " + " | ".join(f"{v:+.6f}" for v in vals) + " |")
    text.extend(
        [
            "",
            "Pair-level comparisons with low gold support have high uncertainty; zero-support recall remains null. Three seeds use the same test set and do not establish statistical significance.",
            "",
            "Historical neural manifest and training-source equivalence remain user-accepted assumptions, not verified provenance.",
        ]
    )
    payloads["test_02b_integrity.json"] = (
        json.dumps(
            {
                "status": "PASS",
                "runs": sources,
                "comparison_tolerance": {"relative": 1e-10, "absolute": 1e-12},
                "historical_provenance": "assumed equivalence; not verified",
            },
            indent=2,
        )
        + "\n"
    ).encode()
    payloads["test_02b_quantitative_analysis.md"] = ("\n".join(text) + "\n").encode()
    for name, payload in payloads.items():
        target = out / name
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError(f"Existing TEST-02B artifact differs; preserving it: {target}")
    for name, payload in payloads.items():
        write_once(out / name, payload)
    print("TEST-02B quantitative artifacts verified/written", flush=True)
