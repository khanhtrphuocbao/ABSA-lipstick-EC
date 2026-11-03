"""Review-level error analysis of sealed predictions; no model inference."""

import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict

from src.analysis.common import ROOT, csv_bytes as serialize_csv, digest, write_once
from src.analysis.test02b.core import CANDIDATES, SEEDS

# Preserve the frozen evaluator's top-level module identity and imports.
sys.path.insert(0, str(ROOT / "src"))
from absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS, encode_targets, evaluate_rows
from evaluate_predictions import align_predictions
from preflight_validation import check as frozen_preflight

SUBSETS = (
    "original_test",
    "test_nonoverlap_l1",
    "test_nonoverlap_l2",
    "test_without_placeholder",
    "test_placeholder_only",
)
EXPECTED = (1623, 1592, 1585, 1615, 8)


def events_for(gold, pred, candidate, seed):
    target = encode_targets(gold)
    predicted = set(pred["aspects"])
    result = []
    for aspect in CORE_ASPECTS:
        g = gold[aspect]
        detected = aspect in predicted
        sentiment = pred["conditional_sentiments"][aspect]
        if g:
            outcome = (
                "MISSED_ASPECT"
                if not detected
                else (
                    "CORRECT_ASPECT_CORRECT_SENTIMENT"
                    if sentiment == g
                    else "CORRECT_ASPECT_WRONG_SENTIMENT"
                )
            )
            typ = "GOLD_PRESENT"
        elif detected:
            outcome, typ = "SPURIOUS_ASPECT", "SPURIOUS_ASPECT"
        else:
            continue
        result.append(
            dict(
                source_split="test",
                source_row_index=int(gold["source_row_index"]),
                candidate=candidate,
                seed=seed,
                aspect=aspect,
                event_type=typ,
                gold_sentiment=g,
                aspect_detected=detected,
                conditional_sentiment=sentiment,
                outcome=outcome,
            )
        )
    return result


def feature_flags(text, gold):
    lower = text.lower()
    words = lower.split()
    return {
        "negation": bool(re.search(r"\b(?:không|chẳng|chưa|ko|k|khong)\b", lower)),
        "contrast": bool(re.search(r"\b(?:nhưng|mà|tuy|nhung)\b", lower)),
        "emoji_or_emoticon": bool(re.search(r"[\U0001F300-\U0001FAFF]|[:;=][-^]?[)(D]", text)),
        "repeated_letters": bool(re.search(r"([^\W\d_])\1{2,}", lower)),
        "short_le_5_whitespace_words": len(words) <= 5,
        "long_ge_100_whitespace_words": len(words) >= 100,
        "multiple_core_aspects": sum(bool(gold[a]) for a in CORE_ASPECTS) >= 2,
        "mixed_gold_sentiment": len({gold[a] for a in CORE_ASPECTS if gold[a]}) >= 2,
    }


def csv_bytes(rows, fields):
    """Keep TEST-02C's historical extra-field projection behavior."""
    return serialize_csv(rows, fields, extrasaction="ignore")


def event_confusion(ev, gold, check, candidate, seed):
    """Reconcile Joint counts, then distinguish conditional and gated confusion."""
    rid = f"{candidate}-seed-{seed}"
    confusion = []
    for a in CORE_ASPECTS:
        for s in SENTIMENTS:
            tp = sum(
                e["aspect"] == a
                and e["gold_sentiment"] == s
                and e["outcome"] == "CORRECT_ASPECT_CORRECT_SENTIMENT"
                for e in ev
            )
            fn = sum(
                e["aspect"] == a
                and e["gold_sentiment"] == s
                and e["outcome"] in ("MISSED_ASPECT", "CORRECT_ASPECT_WRONG_SENTIMENT")
                for e in ev
            )
            fp = sum(
                e["aspect"] == a
                and e["conditional_sentiment"] == s
                and e["outcome"] in ("SPURIOUS_ASPECT", "CORRECT_ASPECT_WRONG_SENTIMENT")
                for e in ev
            )
            actual = check["joint"]["per_label"][f"{a}:{s}"]
            if (tp, fp, fn) != (actual["tp"], actual["fp"], actual["fn"]):
                raise ValueError(f"pair reconciliation {rid} {a}:{s}")
        for gs in SENTIMENTS:
            selected = [
                e
                for e in ev
                if e["aspect"] == a
                and e["gold_sentiment"] == gs
                and e["event_type"] == "GOLD_PRESENT"
            ]
            support = len(selected)
            for definition in ("conditional", "detection_gated"):
                for ps in (*SENTIMENTS, "MISSED"):
                    if definition == "conditional" and ps == "MISSED":
                        continue
                    n = sum(
                        (
                            e["conditional_sentiment"]
                            if definition == "conditional" or e["aspect_detected"]
                            else "MISSED"
                        )
                        == ps
                        for e in selected
                    )
                    confusion.append(
                        dict(
                            candidate=candidate,
                            seed=seed,
                            aspect=a,
                            definition=definition,
                            gold_sentiment=gs,
                            predicted_sentiment=ps,
                            count=n,
                            denominator=support,
                            rate=n / support if support else None,
                            gold_support=support,
                        )
                    )
        absent = sum(not g[a] for g in gold)
        n = sum(e["aspect"] == a and e["outcome"] == "SPURIOUS_ASPECT" for e in ev)
        confusion.append(
            dict(
                candidate=candidate,
                seed=seed,
                aspect=a,
                definition="detection",
                gold_sentiment="ABSENT",
                predicted_sentiment="SPURIOUS",
                count=n,
                denominator=absent,
                rate=n / absent if absent else None,
                gold_support=0,
            )
        )
    return confusion


def subset_diagnostics(report, gold, ordered, masks, candidate, seed):
    """Project the same predictions onto the five frozen subset masks."""
    rid = f"{candidate}-seed-{seed}"
    diagnostics = []
    for subset in SUBSETS:
        m = report["reports"][subset]
        idx = masks[subset]
        if m["joint"]["reviews"] != len(idx):
            raise ValueError(f"subset mismatch {rid} {subset}")
        projected = evaluate_rows([gold[i] for i in idx], [ordered[i] for i in idx])
        if projected != m:
            raise ValueError(f"diagnostic mismatch {rid} {subset}")
        diagnostics.append(
            dict(
                candidate=candidate,
                seed=seed,
                subset=subset,
                reviews=len(idx),
                joint_macro_f1=m["joint"]["macro_f1"],
                joint_micro_f1=m["joint"]["micro"]["f1"],
                others_tp=m["others"]["tp"],
                others_fp=m["others"]["fp"],
                others_fn=m["others"]["fn"],
                others_f1=m["others"]["f1"],
                core_fp=sum(x["fp"] for x in m["joint"]["per_label"].values()),
                label_support=json.dumps(
                    {k: v["support"] for k, v in m["joint"]["per_label"].items()},
                    sort_keys=True,
                ),
            )
        )
    return diagnostics


def collect_runs(manifest, gold, masks):
    """Read sealed predictions and collect validated per-run analysis rows."""
    runs = ROOT / "absa-runs-final-test"
    frozen = ROOT / "reports/experiments/test_01/final_selection_freeze.json"
    test = ROOT / "data/data_test.csv"
    protocol = ROOT / "protocols/evaluation_protocol_v1.yaml"
    evaluator = ROOT / "src/evaluate_predictions.py"
    all_events = []
    preds = {}
    runhash = {}
    confusion = []
    diagnostics = []
    for candidate in CANDIDATES:
        for seed in SEEDS:
            rid = f"{candidate}-seed-{seed}"
            d = runs / rid / "final_test"
            paths = [
                d / "test_predictions.jsonl",
                d / "test_metrics.json",
                d / "inference_manifest.json",
                d / "test_predictions.sha256",
            ]
            predpath, metricpath, impath, side = paths
            im = json.loads(impath.read_text())
            report = json.loads(metricpath.read_text())
            if (
                im["run_id"] != rid
                or digest(predpath) != im["prediction_sha256"]
                or digest(predpath) != side.read_text().strip()
                or digest(metricpath) != im["metrics_sha256"]
                or digest(test) != im["test_sha256"]
                or digest(protocol) != im["protocol_sha256"]
                or digest(evaluator) != im["evaluator_sha256"]
                or digest(frozen) != im["freeze_sha256"]
            ):
                raise ValueError(f"integrity mismatch: {rid}")
            raw = [json.loads(line) for line in predpath.read_text().splitlines() if line.strip()]
            aligned_gold, ordered = align_predictions(manifest, raw, "test")
            if len(ordered) != 1623:
                raise ValueError(f"coverage {rid}")
            check = evaluate_rows(aligned_gold, ordered)
            if check != report["reports"]["original_test"]:
                raise ValueError(f"evaluator mismatch {rid}")
            preds[candidate, seed] = ordered
            runhash[rid] = {
                "prediction": digest(predpath),
                "metrics": digest(metricpath),
                "manifest": digest(impath),
            }
            ev = [e for g, p in zip(gold, ordered) for e in events_for(g, p, candidate, seed)]
            all_events.extend(ev)
            confusion.extend(event_confusion(ev, gold, check, candidate, seed))
            diagnostics.extend(subset_diagnostics(report, gold, ordered, masks, candidate, seed))
            print(rid, "verified and reconciled", flush=True)
    return all_events, preds, runhash, confusion, diagnostics


def compare_reviews(gold, preds, all_events):
    """Align the same review and aspect across models, keeping seed 42 sampling separate."""
    indexed = {
        (e["candidate"], e["seed"], e["source_row_index"], e["aspect"]): e
        for e in all_events
        if e["event_type"] == "GOLD_PRESENT"
    }
    cross = []
    samples = defaultdict(list)
    features = []
    for seed in SEEDS:
        for i, g in enumerate(gold):
            flags = feature_flags(g["raw_text"], g)
            for a in CORE_ASPECTS:
                if not g[a]:
                    if seed == 42 and any(a in preds[c, seed][i]["aspects"] for c in CANDIDATES):
                        samples["spurious_aspect"].append(
                            (int(g["source_row_index"]), a, i, ["spurious_aspect"])
                        )
                    continue
                es = {c: indexed[c, seed, int(g["source_row_index"]), a] for c in CANDIDATES}
                row = dict(
                    source_split="test",
                    source_row_index=int(g["source_row_index"]),
                    aspect=a,
                    gold_sentiment=g[a],
                    seed=seed,
                    **{c.replace("-", "_"): es[c]["outcome"] for c in CANDIDATES},
                )
                cross.append(row)
                for c in CANDIDATES:
                    for flag, value in flags.items():
                        features.append(
                            (
                                c,
                                seed,
                                flag,
                                value,
                                es[c]["outcome"] != "CORRECT_ASPECT_CORRECT_SENTIMENT",
                            )
                        )
                if seed == 42:
                    correct = lambda c: es[c]["outcome"] == "CORRECT_ASPECT_CORRECT_SENTIMENT"
                    categories = []
                    if g[a] == "negative" and correct("PHOBERT-02") and not correct("BIGRU-02"):
                        categories.append("phobert_negative_correct_bigru_wrong")
                    if g[a] == "negative" and correct("PHOBERT-02") and not correct("TFIDF-02"):
                        categories.append("phobert_negative_correct_tfidf_wrong")
                    if g[a] == "neutral" and correct("BIGRU-02") and not correct("PHOBERT-02"):
                        categories.append("bigru_neutral_correct_phobert_wrong")
                    if all(not correct(c) for c in CANDIDATES):
                        categories.append("all_models_wrong")
                    if any(
                        es[c]["outcome"] == "CORRECT_ASPECT_WRONG_SENTIMENT"
                        and es[c]["conditional_sentiment"] == "positive"
                        for c in CANDIDATES
                    ) and g[a] in ("neutral", "negative"):
                        categories.append(g[a] + "_to_positive")
                    if any(es[c]["outcome"] == "MISSED_ASPECT" for c in CANDIDATES):
                        categories.append("missed_aspect")
                    if flags["mixed_gold_sentiment"]:
                        categories.append("mixed_gold_sentiments")
                    if (
                        flags["short_le_5_whitespace_words"]
                        or flags["repeated_letters"]
                        or flags["emoji_or_emoticon"]
                    ) and any(not correct(c) for c in CANDIDATES):
                        categories.append("short_or_noisy_with_error")
                    for category in categories:
                        samples[category].append((int(g["source_row_index"]), a, i, categories))
    return indexed, cross, samples, features


def summarize_features(features, gold):
    """Count observed text features without assigning causal explanations."""
    linguistic = []
    for c in CANDIDATES:
        for seed in SEEDS:
            for flag in feature_flags("", gold[0]):
                for value in (False, True):
                    z = [
                        v
                        for cc, ss, ff, v, err in features
                        if cc == c and ss == seed and ff == flag and v == value
                    ]
                    errors = sum(
                        err
                        for cc, ss, ff, v, err in features
                        if cc == c and ss == seed and ff == flag and v == value
                    )
                    linguistic.append(
                        dict(
                            candidate=c,
                            seed=seed,
                            feature=flag,
                            feature_present=value,
                            unit="gold_present_review_aspect",
                            count=len(z),
                            error_count=errors,
                            error_rate=errors / len(z) if z else None,
                        )
                    )
    return linguistic


def sample_examples(samples, preds, indexed, gold):
    """Select the first five eligible cases per category, before privacy review."""
    examples = []
    for category, items in sorted(samples.items()):
        for rank, (idx, a, i, categories) in enumerate(sorted(items)[:5], 1):
            for c in CANDIDATES:
                p = preds[c, 42][i]
                e = indexed.get((c, 42, idx, a))
                outcome = (
                    e["outcome"]
                    if e
                    else ("SPURIOUS_ASPECT" if a in p["aspects"] else "CORRECT_ABSENCE")
                )
                examples.append(
                    dict(
                        category=category,
                        population=len(items),
                        sample_rank=rank,
                        source_split="test",
                        source_row_index=idx,
                        aspect=a,
                        gold_sentiment=gold[i][a],
                        candidate=c,
                        seed=42,
                        predicted_aspects="|".join(p["aspects"]),
                        conditional_sentiment=p["conditional_sentiments"][a],
                        outcome=outcome,
                        category_overlap="|".join(categories),
                        review_excerpt="[withheld pending privacy review]",
                        privacy_status="PENDING_MANUAL_REVIEW",
                    )
                )
    return examples


def write_tables(out, tables):
    """Export numeric tables in their historical column and row order."""
    schemas = {
        "error_taxonomy.csv": [
            "source_split",
            "source_row_index",
            "candidate",
            "seed",
            "aspect",
            "event_type",
            "gold_sentiment",
            "aspect_detected",
            "conditional_sentiment",
            "outcome",
        ],
        "confusion_summary.csv": [
            "candidate",
            "seed",
            "aspect",
            "definition",
            "gold_sentiment",
            "predicted_sentiment",
            "count",
            "denominator",
            "rate",
            "gold_support",
        ],
        "cross_model_error_comparison.csv": [
            "source_split",
            "source_row_index",
            "aspect",
            "gold_sentiment",
            "seed",
            "TFIDF_02",
            "BIGRU_02",
            "PHOBERT_02",
        ],
        "linguistic_error_summary.csv": [
            "candidate",
            "seed",
            "feature",
            "feature_present",
            "unit",
            "count",
            "error_count",
            "error_rate",
        ],
        "diagnostic_subset_comparison.csv": [
            "candidate",
            "seed",
            "subset",
            "reviews",
            "joint_macro_f1",
            "joint_micro_f1",
            "others_tp",
            "others_fp",
            "others_fn",
            "others_f1",
            "core_fp",
            "label_support",
        ],
        "representative_errors.csv": [
            "category",
            "population",
            "sample_rank",
            "source_split",
            "source_row_index",
            "aspect",
            "gold_sentiment",
            "candidate",
            "seed",
            "predicted_aspects",
            "conditional_sentiment",
            "outcome",
            "category_overlap",
            "review_excerpt",
            "privacy_status",
        ],
    }
    for name, fields in schemas.items():
        write_once(out / name, csv_bytes(tables[name], fields))


def write_reports(out, runhash, before, all_events, diagnostics):
    """Keep historical report text and integrity metadata unchanged."""
    integrity = {
        "status": "PARTIAL",
        "run_hashes": runhash,
        "input_sha256": before,
        "limitation": "ASSUMED_MANIFEST_EQUIVALENCE; historical neural provenance not verified",
        "partial": [
            "Review excerpts withheld pending privacy review; potential annotation ambiguity requires manual review.",
            "Complete qualitative review remains to be completed.",
        ],
        "seed_aggregation": "Available from confusion_seed_aggregate.csv and diagnostic_seed_aggregate.csv after running supplement module.",
    }
    write_once(
        out / "test_02c_integrity.json",
        (json.dumps(integrity, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )
    diagnostic = "# Diagnostic subsets\n\nFrozen masks have 1623, 1592, 1585, 1615 and 8 reviews. Fixed-21 support ceilings are 21/21, 20/21, 20/21, 21/21 and 0/21. See diagnostic_subset_comparison.csv for seed metrics, OTHERS, core false positives and per-label support. L2 is aggressive screening; L3 is incomplete. NAN provenance is unresolved. A small score change does not prove absence of leakage.\n"
    write_once(out / "diagnostic_subset_analysis.md", diagnostic.encode())
    report = "# TEST-02C Review-Level Error Analysis — PARTIAL\n\nNine frozen runs were aligned by test row and Joint TP/FP/FN reconciled with sealed evaluator outputs. Confusion counts distinguish gold-present conditional sentiment from detection-gated sentiment; missing aspects have their own gated column. Per-aspect spurious rates use gold-absent review count as denominator. Tables retain each seed; these are repeated model runs on one test set.\n\n## Seed 42 counts\n\n| Candidate | Correct | Wrong sentiment | Missed aspect | Spurious aspect |\n|---|---:|---:|---:|---:|\n"
    for candidate in CANDIDATES:
        counts = Counter(
            e["outcome"] for e in all_events if e["candidate"] == candidate and e["seed"] == 42
        )
        report += f"| {candidate} | {counts['CORRECT_ASPECT_CORRECT_SENTIMENT']} | {counts['CORRECT_ASPECT_WRONG_SENTIMENT']} | {counts['MISSED_ASPECT']} | {counts['SPURIOUS_ASPECT']} |\n"
    report += "\nCounts are review–aspect events, not counts of independent reviews. Seed 42 is descriptive; see confusion_seed_aggregate.csv for all seeds.\n\n## Diagnostic Joint Macro-F1\n\n| Candidate | Original | Nonoverlap L1 | Nonoverlap L2 | Without placeholder | Placeholder only |\n|---|---:|---:|---:|---:|---:|\n"
    for candidate in CANDIDATES:
        vals = [
            statistics.mean(
                r["joint_macro_f1"]
                for r in diagnostics
                if r["candidate"] == candidate and r["subset"] == subset
            )
            for subset in SUBSETS
        ]
        report += "| " + candidate + " | " + " | ".join(f"{v:.6f}" for v in vals) + " |\n"
    report += "\nSubset class support differs; the fixed-21 scores are not ceiling-normalized. Placeholder-only has no core gold support, so its zero is not evidence of sentiment understanding. L2 is aggressive screening; L3 is incomplete and small score shifts do not prove absence of leakage. NAN provenance is unresolved.\n\nText feature flags are descriptive heuristics, not evidence of causation. Whitespace length does not establish token truncation. Review excerpts are withheld pending privacy review and potential annotation ambiguity requires manual assessment. Historical neural manifest equivalence remains assumed, not verified. No inference or fitting was performed. See test_02c_integrity.json for remaining work.\n"
    write_once(out / "test_02c_error_analysis.md", report.encode())


def run_analysis():
    runs = ROOT / "absa-runs-final-test"
    out = ROOT / "reports/experiments/test_02"
    frozen_preflight(ROOT)
    manifest_path = ROOT / "protocols/data_audit_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    gold = [r for r in manifest if r["source_split"] == "test"]
    if len(gold) != 1623:
        raise ValueError("test manifest row count")
    masks = {
        s: [i for i, g in enumerate(gold) if s == "original_test" or g[s] == "True"]
        for s in SUBSETS
    }
    if [len(masks[s]) for s in SUBSETS] != list(EXPECTED):
        raise ValueError("subset coverage mismatch")
    frozen = ROOT / "reports/experiments/test_01/final_selection_freeze.json"
    test = ROOT / "data/data_test.csv"
    protocol = ROOT / "protocols/evaluation_protocol_v1.yaml"
    evaluator = ROOT / "src/evaluate_predictions.py"
    input_files = [
        manifest_path,
        test,
        protocol,
        evaluator,
        ROOT / "src/absa_protocol.py",
        frozen,
        ROOT / "protocols/freeze_lock_v1.json",
        runs / "final_test_summary.json",
        ROOT / "reports/experiments/test_02/test_02b_integrity.json",
        ROOT / "reports/experiments/test_02/per_pair_metrics.csv",
    ]
    for candidate in CANDIDATES:
        for seed in SEEDS:
            directory = runs / f"{candidate}-seed-{seed}" / "final_test"
            input_files.extend(
                directory / name
                for name in (
                    "test_predictions.jsonl",
                    "test_metrics.json",
                    "inference_manifest.json",
                    "test_predictions.sha256",
                )
            )
    before = {str(p.relative_to(ROOT)): digest(p) for p in input_files}
    all_events, preds, runhash, confusion, diagnostics = collect_runs(manifest, gold, masks)
    indexed, cross, samples, features = compare_reviews(gold, preds, all_events)
    linguistic = summarize_features(features, gold)
    examples = sample_examples(samples, preds, indexed, gold)
    write_tables(
        out,
        {
            "error_taxonomy.csv": all_events,
            "confusion_summary.csv": confusion,
            "cross_model_error_comparison.csv": cross,
            "linguistic_error_summary.csv": linguistic,
            "diagnostic_subset_comparison.csv": diagnostics,
            "representative_errors.csv": examples,
        },
    )
    write_reports(out, runhash, before, all_events, diagnostics)
    if before != {str(p.relative_to(ROOT)): digest(p) for p in input_files}:
        raise RuntimeError("upstream mutation detected")
    print("TEST-02C PARTIAL: quantitative artifacts written; see integrity report")
