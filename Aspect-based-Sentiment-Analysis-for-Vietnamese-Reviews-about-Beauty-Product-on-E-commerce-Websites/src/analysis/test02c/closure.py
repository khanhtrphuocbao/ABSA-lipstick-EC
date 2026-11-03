"""Review sampled cases and reconcile previously generated TEST-02C tables."""

import csv, hashlib, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
from src.analysis.test02c.pipeline import ROOT, CORE_ASPECTS, SENTIMENTS, csv_bytes
from src.analysis.common import write_once

OUT = ROOT / "reports/experiments/test_02"
# Manually reviewed short excerpts. None includes a visible name, phone, email,
# address or account identifier; this is a best-effort review, not a guarantee.
EXCERPTS = {
    1: "đóng gói cẩn thận, dịch nhưng mà giao hàng khá nhanh",
    2: "Chất son mềm mướt nói chung là rất thíchhhh",
    7: "Dùng hơi khô nhưng với giá như vậy là ok rồi",
    8: "Chất son lì màu cam siu đẹp nhma mà giao hàng khá lâu",
    11: "Son màu đẹp, mùi không thơm lắm",
    12: "Chất son mịn, lên màu đẹp lắm ý mà hơi khô môi một chút",
    13: "mịn môi,bám lâu, màu bền, đẹp",
    14: "Giao hàng chậm. Shop giao đúng mẫu sp sài ok",
    15: "Đợi mãi cuối cùng cũng về. Mới toanh luôn",
    16: "Son mềm mịn nhưng k đúng tone màu cho lắm",
    18: "Shop giao đủ hàng,đóng gói tốt",
    25: "Son thì chất mềm mịn, dễ tán, đánh lên môi thích cực",
    35: "chất son trên môi k quá khô vẫn còn chút độ ẩm",
    36: "Màu 25 đẹp. Lần này gói hàng k cẩn thận bằng lần trước",
    37: "Chất son lỏng.. Mùi hơi nồng.. Nhưng ổn",
    45: "Giao hàng hơi lâu",
    46: "Đóng gói cẩn thận",
    47: "Màu đẹp.rất hài lòng",
    48: "Màu Rất đẹp giao hàng nhanh hơn mình nghĩ đóng gói cẩn thận",
    52: "Son thì cũng ok khá nhanh trôi",
    54: "shop đóng gói hàng rất kĩ, son cầm rất chắc tay",
    61: "đóng gói sơ sài khiến vỏ son bị móp",
    66: "Kết cấu: rất bột, bệt hết cả môi",
    88: "Màu hơi sậm một chút so với ảnh",
    159: "màu rất đẹp nha nó cũng không khô quá",
    275: "hơi khô 1 xíu",
    518: "minh chua nhan duoc huhu",
}
# Manual flags reflect observed surface forms, not inferred model mechanisms.
FLAGS = {
    "negation": {2, 11, 16, 35, 36, 52, 54, 61, 66, 159, 275, 518},
    "contrast": {1, 2, 7, 8, 16, 25, 35, 37, 45, 46, 52, 54, 61, 88},
    "emoji_emoticon": {45, 48, 52, 54, 275},
    "slang_informal": {7, 8, 12, 13, 14, 16, 25, 35, 37, 45, 52, 61, 66, 275, 518},
    "mixed_sentiment": {7, 8, 11, 12, 14, 16, 25, 35, 36, 45, 46, 52, 54, 61, 66, 88, 275},
}


def read(name):
    with (OUT / name).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    integrity = json.loads((OUT / "test_02c_integrity.json").read_text())
    inputs = integrity["input_sha256"]
    for name, h in inputs.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != h:
            raise ValueError("input drift " + name)
    rows = read("representative_errors.csv")
    slots = {(r["category"], r["sample_rank"]) for r in rows}
    ids = {int(r["source_row_index"]) for r in rows}
    if len(slots) != 50 or ids != set(EXCERPTS):
        raise ValueError("sample set changed; manual review invalid")
    with (ROOT / "protocols/data_audit_manifest.csv").open(encoding="utf8", newline="") as f:
        gold = {
            int(r["source_row_index"]): r for r in csv.DictReader(f) if r["source_split"] == "test"
        }
    for i, text in EXCERPTS.items():
        if text not in gold[i]["raw_text"]:
            raise ValueError(f"excerpt mismatch {i}")
    reviewed = []
    linguistic = []
    for r in rows:
        i = int(r["source_row_index"])
        r = dict(r)
        r["review_excerpt"] = EXCERPTS[i]
        r["privacy_status"] = "REVIEWED_NO_VISIBLE_IDENTIFIER"
        r["interpretation"] = (
            "Gold "
            + (r["gold_sentiment"] or "absent")
            + "; "
            + r["candidate"]
            + " predicts "
            + r["conditional_sentiment"]
            + "; outcome "
            + r["outcome"]
            + "."
        )
        reviewed.append(r)
    for i in sorted(ids):
        relevant = [r for r in reviewed if int(r["source_row_index"]) == i]
        linguistic.append(
            {
                "source_split": "test",
                "source_row_index": i,
                "review_excerpt": EXCERPTS[i],
                **{k: (i in v) for k, v in FLAGS.items()},
                "observation": "Text features are descriptive; no causal attribution. Gold disagreement alone does not establish annotation error.",
                "category_memberships": "|".join(sorted({r["category"] for r in relevant})),
            }
        )
    write_once(OUT / "representative_errors_reviewed.csv", csv_bytes(reviewed, list(reviewed[0])))
    write_once(OUT / "linguistic_manual_review.csv", csv_bytes(linguistic, list(linguistic[0])))
    # Reconcile each run's gold-present event totals and per-label TP/FP/FN.
    events = read("error_taxonomy.csv")
    byrun = defaultdict(list)
    for e in events:
        byrun[e["candidate"], int(e["seed"])].append(e)
    if len(byrun) != 9:
        raise ValueError("expected nine runs")
    per = read("per_pair_metrics.csv")
    anchors = {(r["candidate"], int(r["seed"]), r["label"]): r for r in per}
    for key, ev in byrun.items():
        gold_events = [e for e in ev if e["event_type"] == "GOLD_PRESENT"]
        if (
            len(gold_events) != 2969
            or len({(e["source_row_index"], e["aspect"]) for e in gold_events}) != 2969
        ):
            raise ValueError("gold event coverage " + str(key))
        for a in CORE_ASPECTS:
            for s in SENTIMENTS:
                target = anchors[key[0], key[1], a + ":" + s]
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
                if (tp, fp, fn) != tuple(int(target[k]) for k in ("tp", "fp", "fn")):
                    raise ValueError("pair mismatch " + str(key))
    # Reconcile every confusion cell, including gated misses and spurious rates.
    for r in read("confusion_summary.csv"):
        ev = byrun[r["candidate"], int(r["seed"])]
        a = r["aspect"]
        g = r["gold_sentiment"]
        p = r["predicted_sentiment"]
        if r["definition"] == "detection":
            selected = [e for e in ev if e["aspect"] == a and e["outcome"] == "SPURIOUS_ASPECT"]
            denominator = 1623 - sum(
                e["aspect"] == a and e["event_type"] == "GOLD_PRESENT" for e in ev
            )
            count = len(selected)
        else:
            selected = [
                e
                for e in ev
                if e["aspect"] == a
                and e["event_type"] == "GOLD_PRESENT"
                and e["gold_sentiment"] == g
            ]
            denominator = len(selected)
            count = sum(
                (
                    e["conditional_sentiment"]
                    if r["definition"] == "conditional" or e["aspect_detected"] == "True"
                    else "MISSED"
                )
                == p
                for e in selected
            )
        if count != int(r["count"]) or denominator != int(r["denominator"]):
            raise ValueError("confusion cell mismatch")
        if denominator and not math.isclose(float(r["rate"]), count / denominator, abs_tol=1e-12):
            raise ValueError("confusion rate mismatch")
        if not denominator and r["rate"] != "":
            raise ValueError("zero denominator must be null")
    # Pairing must be aligned to event taxonomy rather than merely matching row counts.
    lookup = {
        (e["candidate"], int(e["seed"]), e["source_row_index"], e["aspect"]): e
        for e in events
        if e["event_type"] == "GOLD_PRESENT"
    }
    cross = read("cross_model_error_comparison.csv")
    if len(cross) != 2969 * 3:
        raise ValueError("cross model row count")
    for r in cross:
        for c in ("TFIDF-02", "BIGRU-02", "PHOBERT-02"):
            e = lookup[c, int(r["seed"]), r["source_row_index"], r["aspect"]]
            if (
                e["outcome"] != r[c.replace("-", "_")]
                or e["gold_sentiment"] != r["gold_sentiment"]
            ):
                raise ValueError("cross pairing mismatch")
    cross_counts = Counter()
    correct = "CORRECT_ASPECT_CORRECT_SENTIMENT"
    for r in cross:
        checks = {
            "negative_phobert_correct_bigru_wrong": r["gold_sentiment"] == "negative"
            and r["PHOBERT_02"] == correct
            and r["BIGRU_02"] != correct,
            "negative_phobert_correct_tfidf_wrong": r["gold_sentiment"] == "negative"
            and r["PHOBERT_02"] == correct
            and r["TFIDF_02"] != correct,
            "neutral_bigru_correct_phobert_wrong": r["gold_sentiment"] == "neutral"
            and r["BIGRU_02"] == correct
            and r["PHOBERT_02"] != correct,
            "all_models_wrong": all(
                r[c] != correct for c in ("TFIDF_02", "BIGRU_02", "PHOBERT_02")
            ),
        }
        for name, match in checks.items():
            cross_counts[name, r["aspect"], int(r["seed"])] += int(match)
    for r in read("cross_model_seed_aggregate.csv"):
        for seed in (42, 2026, 3407):
            if int(r[f"seed{seed}"]) != cross_counts[r["comparison"], r["aspect"], seed]:
                raise ValueError("cross model aggregate mismatch")
    diagnostics = read("diagnostic_subset_comparison.csv")
    ds = defaultdict(list)
    for r in diagnostics:
        ds[r["candidate"], r["subset"]].append(r)
    expected = {
        "original_test": (1623, 21),
        "test_nonoverlap_l1": (1592, 20),
        "test_nonoverlap_l2": (1585, 20),
        "test_without_placeholder": (1615, 21),
        "test_placeholder_only": (8, 0),
    }
    if len(diagnostics) != 45:
        raise ValueError("diagnostic count")
    for r in diagnostics:
        run = f"{r['candidate']}-seed-{r['seed']}"
        actual = json.loads(
            (ROOT / "absa-runs-final-test" / run / "final_test/test_metrics.json").read_text()
        )["reports"][r["subset"]]
        support = json.loads(r["label_support"])
        n, positive = expected[r["subset"]]
        if (
            int(r["reviews"]) != n
            or sum(v > 0 for v in support.values()) != positive
            or any(support[k] != v["support"] for k, v in actual["joint"]["per_label"].items())
        ):
            raise ValueError("subset support mismatch")
        if not math.isclose(
            float(r["joint_macro_f1"]), actual["joint"]["macro_f1"], abs_tol=1e-12
        ):
            raise ValueError("subset metric mismatch")
        if (int(r["others_tp"]), int(r["others_fp"]), int(r["others_fn"])) != tuple(
            actual["others"][k] for k in ("tp", "fp", "fn")
        ):
            raise ValueError("others mismatch")
        if int(r["core_fp"]) != sum(v["fp"] for v in actual["joint"]["per_label"].values()):
            raise ValueError("core fp mismatch")
    text = [
        "# TEST-02C closure and diagnostic review",
        "",
        "50 category sample slots correspond to 34 review–aspect pairs and 27 unique reviews. Each excerpt was checked against the frozen audit manifest and reviewed for visible personal identifiers. This review cannot guarantee anonymization. Each review has three seed-42 model records; no model rationale is inferred.",
        "",
        "Manual flags describe visible text; negation, contrast, informal spelling, emoji and mixed language do not establish why a model erred. Gold/model disagreement is not proof of incorrect annotation. Detailed observations and candidate memberships are in linguistic_manual_review.csv.",
        "",
        "## Diagnostic subsets",
        "",
        "| Candidate | Subset | n | Joint Macro-F1 mean | OTHERS TP/FP/FN per seed | Core FP per seed | Supported labels |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    for (candidate, subset), rs in sorted(ds.items()):
        rs = sorted(rs, key=lambda x: int(x["seed"]))
        mean = statistics.mean(float(r["joint_macro_f1"]) for r in rs)
        oo = ", ".join(
            f"{r['seed']}: {r['others_tp']}/{r['others_fp']}/{r['others_fn']}" for r in rs
        )
        fp = ", ".join(f"{r['seed']}: {r['core_fp']}" for r in rs)
        text.append(
            f'| {candidate} | {subset} | {rs[0]["reviews"]} | {mean:.6f} | {oo} | {fp} | {expected[subset][1]}/21 |'
        )
    text += [
        "",
        "Per-label support is recorded in diagnostic_subset_comparison.csv; no ceiling normalization is applied. Placeholder-only has no core gold support, so zero Joint Macro-F1 is not a measure of sentiment understanding. L2 collisions are screening candidates, L3 remains incomplete, and small metric shifts cannot establish absence of leakage. Literal NAN provenance remains unresolved.",
        "",
        "Historical neural manifest equivalence is assumed by user decision, not verified. Three seeds share the same test dataset; no significance claim is made. No model training, inference, threshold change or test-based model selection was performed.",
    ]
    write_once(OUT / "test_02c_closure_report.md", ("\n".join(text) + "\n").encode())
    result = {
        "status": "REVIEWED_QUANTITATIVE_CHECKS_PASS",
        "sample_slots": len(slots),
        "unique_reviews": len(ids),
        "gold_present_per_run": 2969,
        "runs": 9,
        "test01_test02b_input_hashes_unchanged": True,
        "provenance": "ASSUMED_MANIFEST_EQUIVALENCE; historical provenance not verified",
        "limitations": [
            "Privacy review is best effort; no claim of guaranteed anonymization.",
            "Manual linguistic flags are descriptive and do not establish causation.",
        ],
    }
    write_once(
        OUT / "test_02c_closure_integrity.json",
        (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
