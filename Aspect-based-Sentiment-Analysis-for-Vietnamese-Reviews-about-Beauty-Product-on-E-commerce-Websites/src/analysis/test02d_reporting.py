"""Prepare research manuscripts from verified evidence; no models are loaded."""

import argparse, csv, json, re, statistics, sys
from pathlib import Path
from collections import Counter
from src.analysis.test02d.sources import (
    ROOT,
    OUT,
    CANDIDATES,
    SEEDS,
    ASPECTS,
    SENTIMENTS,
    SUBSETS,
    sha,
    load,
)
from src.analysis.figure_status import stale
from src.analysis.common import write_once

VAL = ROOT.parent / "absa-run/validation_aggregate_assumed_equivalence.json"
OUTPUTS = {
    "results_and_discussion.md",
    "threats_to_validity.md",
    "test_02d_summary.md",
    "test_02d_claims.json",
    "test_02d_integrity.json",
    "test_02d_closure_report.md",
    "test_02d_completion.json",
}


def packed(x):
    return (
        json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def snapshot():
    paths = []
    for name in ("data", "protocols", "absa-runs-final-test", "reports/experiments/test_01"):
        paths.extend(p for p in (ROOT / name).rglob("*") if p.is_file())
    paths.extend(p for p in OUT.rglob("*") if p.is_file() and p.name not in OUTPUTS)
    paths.extend((ROOT / "src/absa_protocol.py", ROOT / "src/evaluate_predictions.py"))
    return {str(p): sha(p) for p in sorted(set(paths))}


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def documents(d):
    claims = {}

    def value(key, values, source, field, scale=1):
        if len(values) != 3:
            raise ValueError("requires three seeds")
        mean = statistics.mean(values) * scale
        sd = statistics.stdev(values) * scale
        claims[key] = {
            "values": values,
            "mean": mean,
            "sample_sd": sd,
            "scale": scale,
            "source": source,
            "field": field,
            "seeds": list(SEEDS),
        }
        return f"{mean:.6f} ± {sd:.6f}" if scale == 1 else f"{mean:.2f} ± {sd:.2f}"

    manifest = json.loads((OUT / "figure_manifest.json").read_text())
    if stale(manifest) or any(f["status"] != "READY" for f in manifest["figures"]):
        raise ValueError("figures stale or blocked")

    def figure(num):
        f = manifest["figures"][num - 1]
        return f"\n\n![{f['figure_id']}: {f['title']}](figures/{Path(f['outputs'][0]['path']).name})\n\n**{f['figure_id']}.** {f['caption']}\n"

    threatrows = [
        (
            "Internal / provenance",
            "Sáu neural manifest links không khớp hash lịch sử",
            "Không xác minh đầy đủ training provenance",
            "Hash prediction/checkpoint và evaluator checks; user acceptance",
            "ASSUMED_MANIFEST_EQUIVALENCE vẫn là assumption",
            "[freeze](../test_01/final_selection_freeze.json)",
        ),
        (
            "Internal / runtime và source",
            "Training-source fingerprint khác lịch sử; local CPU khác Colab",
            "Không chứng minh tái lập training tuyệt đối",
            "Ghi source/runtime và user acceptance",
            "Arithmetic verification không xác minh source equivalence",
            "[execution report](../test_01/test_01_execution_report.md)",
        ),
        (
            "Statistical conclusion",
            "Ba seeds dùng cùng test set",
            "SD không phản ánh toàn bộ sampling uncertainty",
            "Giữ từng seed; sample SD ddof=1",
            "Không phải ba independent test datasets; không significance claim",
            "[test summary](../../../absa-runs-final-test/final_test_summary.json)",
        ),
        (
            "Deterministic baseline",
            "TF-IDF outputs giống nhau giữa seeds",
            "Zero SD dễ bị hiểu sai",
            "Công bố identical predictions",
            "Không ước lượng data uncertainty",
            "[model table](model_comparison.csv)",
        ),
        (
            "Rare labels",
            "Một số Neutral/Negative pairs ít support",
            "F1 nhạy với vài review",
            "Báo support, giữ mọi labels",
            "Không suy rộng chắc chắn từ rare labels",
            "[pair table](per_pair_metrics.csv)",
        ),
        (
            "Construct / zero support",
            "Fixed-21 gồm low/zero-support labels ở subsets",
            "Macro khác Micro và pooled sentiment; ceiling khác nhau",
            "Giữ fixed denominator và null recall convention",
            "Không ceiling-normalize; metrics không thay thế mục tiêu nghiệp vụ",
            "[protocol](../../../protocols/evaluation_protocol_v1.yaml)",
        ),
        (
            "Overlap L2",
            "L2 collision screening mạnh",
            "Có thể loại review không phải semantic duplicate",
            "Báo L1/L2 riêng và support changes",
            "Không xác nhận mọi collision là duplicate",
            "[diagnostics](diagnostic_subset_comparison.csv)",
        ),
        (
            "Contamination / L3",
            "Near-duplicate L3 chưa đầy đủ",
            "Có thể còn overlap chưa phát hiện",
            "Công bố screening limits",
            "Small score shifts không chứng minh absence of leakage",
            "[audit summary](../../../reports/protocol_freeze/audit_summary_v1.json)",
        ),
        (
            "Data quality / NAN",
            "Nguồn gốc 105 literal NAN chưa rõ",
            "Placeholder gây khó diễn giải",
            "Frozen placeholder diagnostics; không đổi nhãn",
            "Không tự coi NAN là confirmed missing",
            "[NAN audit](../../../reports/protocol_freeze/NAN_normalization_audit_v1.md)",
        ),
        (
            "Qualitative sampling",
            "50 category slots, 27 reviews, 34 review–aspect pairs; seed 42",
            "Mẫu có chủ đích và chồng lấn",
            "Công bố IDs, categories và seed",
            "Không đại diện population; không ước lượng prevalence",
            "[examples](representative_errors_reviewed.csv)",
        ),
        (
            "Annotation ambiguity",
            "Gold/model disagreements cần đọc lại",
            "Gold không phải unquestionable semantic truth",
            "Giữ gold, ghi ambiguity",
            "Disagreement không chứng minh gold sai",
            "[manual review](test_02c_linguistic_manual.md)",
        ),
        (
            "Privacy",
            "Manual review cho visible identifiers",
            "Có thể còn identifier gián tiếp",
            "Chỉ dùng reviewed excerpts",
            "Không bảo đảm full anonymization",
            "[examples](representative_errors_reviewed.csv)",
        ),
        (
            "Post-hoc interpretation",
            "Heuristic và manual linguistic flags",
            "Association không phải causal mechanism",
            "Tách observation khỏi hypothesis",
            "Không suy attention/internal reasoning",
            "[linguistic review](linguistic_manual_review.csv)",
        ),
        (
            "External / business",
            "Domain son môi Shopee; chưa deployment",
            "Domain drift, costs và workflow khác",
            "Human oversight và independent validation được đề xuất",
            "Không đo ROI, sales, retention, returns, satisfaction, seller performance",
            "[business](business_interpretation.md)",
        ),
    ]
    threats = (
        "# Threats to Validity — TEST-02D.4\n\nMa trận ghi observed issues, potential effects, mitigation và remaining limitations. Potential effect là rủi ro, không phải kết quả nhân quả đã được chứng minh.\n\n"
        + table(
            [
                "Threat category",
                "Observed issue",
                "Potential effect",
                "Existing mitigation",
                "Remaining limitation",
                "Evidence",
            ],
            threatrows,
        )
        + "\n\nKhông sửa data/protocol/model hoặc provenance lịch sử để xử lý các giới hạn này. Kết luận chỉ áp dụng cho evidence được dẫn.\n"
    )
    text = """# Results & Discussion — Vietnamese ABSA on Shopee Lipstick Reviews

## 1. Experimental Evaluation Overview

Nghiên cứu so sánh TFIDF-02, BIGRU-02 và PHOBERT-02 đã chọn bằng validation, với seeds 42, 2026, 3407. Official test có 1.623 review; train/validation/test lần lượt 12.981/1.623/1.623 từ tổng 16.227 review. Protocol frozen quy định tám aspect detection labels; bảy core aspects và ba sentiment classes tạo 21 core pairs. OTHERS là binary detection target riêng. Primary metric là fixed-21 Joint Macro-F1; Joint Micro-F1, Weighted-F1 và observed-support Macro-F1 được phân biệt.

Các bảng mean ± SD dùng arithmetic mean và sample SD (ddof=1) trên ba seed cùng test set; không tính F1 từ averaged counts. SD không phải confidence interval. Candidate, checkpoint và threshold giữ nguyên sau validation selection. Evaluation diễn ra dưới assumed-equivalence exception cho sáu neural manifest links và khác biệt source fingerprint lịch sử; đây không phải fully verified training provenance. Nguồn: [protocol](../../../protocols/evaluation_protocol_v1.yaml), [freeze](../test_01/final_selection_freeze.json), [final execution report](../test_01/test_01_execution_report.md).

## 2. Overall Model Performance

"""
    rows = []
    for c in CANDIDATES:
        rows.append(
            [c]
            + [
                value(
                    c + "." + k,
                    [float(d["model"][c, str(s)][k]) for s in SEEDS],
                    "model_comparison.csv",
                    k,
                )
                for k in (
                    "joint_macro_f1",
                    "joint_micro_f1",
                    "joint_weighted_f1",
                    "observed_support_macro_f1",
                )
            ]
        )
    text += table(
        [
            "Candidate",
            "Joint Macro-F1 fixed21",
            "Joint Micro-F1",
            "Joint Weighted-F1",
            "Observed-support Macro-F1",
        ],
        rows,
    )
    text += (
        "\n\nNguồn: [model table](model_comparison.csv), đối chiếu [TEST-01 summary](../../../absa-runs-final-test/final_test_summary.json). PhoBERT có mean Joint Macro-F1 cao nhất trong các run quan sát. TF-IDF có Micro-F1 cao hơn BiGRU dù Macro-F1 thấp hơn, cho thấy thứ tự phụ thuộc metric. Trên original test mọi pair có gold support nên observed-support Macro-F1 bằng fixed-21; điều này không đúng ở mọi diagnostic subset. Không có significance claim; zero SD TF-IDF không đo sampling uncertainty.\n"
        + figure(1)
    )
    text += "\n## 3. Validation vs Test\n\n"
    rows = []
    for c in CANDIDATES:
        rows.append(
            [
                c,
                value(
                    c + ".validation",
                    d["validation"]["values"][c],
                    "../test_01/final_selection_freeze.json",
                    "runs[].validation_macro_f1",
                ),
                value(
                    c + ".test",
                    [float(d["model"][c, str(s)]["joint_macro_f1"]) for s in SEEDS],
                    "model_comparison.csv",
                    "joint_macro_f1",
                ),
            ]
        )
    text += table(["Candidate", "Validation Joint Macro-F1", "Test Joint Macro-F1"], rows)
    text += (
        "\n\nValidation aggregate có hash khớp selection freeze; historical equivalence vẫn là assumption. Chênh lệch validation/test mô tả hai splits. Composition hoặc class support có thể liên quan, nhưng aggregate scores không xác định nguyên nhân. Kết quả không tạo một model selection mới.\n"
        + figure(2)
    )
    text += "\n## 4. Minority Sentiment Performance\n\n"
    rows = []
    for sentiment in SENTIMENTS:
        for c in CANDIDATES:
            rows.append(
                [sentiment, c]
                + [
                    value(
                        c + "." + sentiment + "." + k,
                        [float(d["sentiment"][c, str(s), sentiment][k]) for s in SEEDS],
                        "sentiment_comparison.csv",
                        sentiment + "." + k,
                        100,
                    )
                    for k in ("precision", "recall", "f1")
                ]
                + [d["sentiment"][c, "42", sentiment]["support"]]
            )
    text += table(["Sentiment", "Candidate", "P (%)", "R (%)", "F1 (%)", "Gold support"], rows)
    text += (
        "\n\nNguồn: [pooled sentiment table](sentiment_comparison.csv). Joint pooled-per-sentiment đòi hỏi đúng aspect và sentiment. PhoBERT có Negative Recall cao hơn cả hai candidates; Precision cao hơn BiGRU nhưng thấp hơn TF-IDF. BiGRU có Neutral Recall cao hơn PhoBERT, trong khi PhoBERT có Neutral Precision/F1 cao hơn. Micro-F1 chịu ảnh hưởng phân bố nhãn; không thay thế minority analysis.\n"
        + figure(3)
    )
    rows = []
    for a, b in [("PHOBERT-02", "BIGRU-02"), ("PHOBERT-02", "TFIDF-02"), ("BIGRU-02", "TFIDF-02")]:
        rows.append(
            [a + " minus " + b]
            + [
                value(
                    a + "-" + b + "." + sentiment,
                    [
                        float(
                            d["decomposition"][a + " minus " + b, str(s), sentiment][
                                "fixed21_contribution"
                            ]
                        )
                        for s in SEEDS
                    ],
                    "macro_f1_decomposition.csv",
                    sentiment + ".fixed21_contribution",
                    100,
                )
                for sentiment in SENTIMENTS
            ]
        )
    text += "\n### Fixed-21 decomposition\n\n" + table(
        [
            "Comparison",
            "Positive contribution (pp)",
            "Neutral contribution (pp)",
            "Negative contribution (pp)",
        ],
        rows,
    )
    text += (
        "\n\nContribution bằng tổng bảy pair-F1 differences chia 21, tính từng seed rồi tổng hợp. Tổng ba groups khớp overall Macro-F1 delta trong tolerance 1e-12. Negative là contribution lớn nhất của PhoBERT–BiGRU; Neutral lớn nhất của BiGRU–TF-IDF. Pooled Negative F1 không phải contribution này. Nguồn: [decomposition](macro_f1_decomposition.csv).\n"
        + figure(5)
    )
    text += "\n## 5. Per-Aspect and Per-Pair Performance\n\n"
    rows = []
    for a in ASPECTS:
        rows.append(
            [a, d["pairs"]["TFIDF-02", "42", a + ":negative"]["support"]]
            + [
                value(
                    c + "." + a + ".negative",
                    [float(d["pairs"][c, str(s), a + ":negative"]["f1"]) for s in SEEDS],
                    "per_pair_metrics.csv",
                    a + ":negative.f1",
                    100,
                )
                for c in CANDIDATES
            ]
        )
    text += table(
        ["Aspect", "Negative support", "TFIDF-02 F1 (%)", "BIGRU-02 F1 (%)", "PHOBERT-02 F1 (%)"],
        rows,
    )
    text += (
        "\n\nNguồn: [pair metrics](per_pair_metrics.csv) và [paired deltas](per_pair_deltas.csv). PhoBERT cải thiện COLOUR/TEXTURE Negative; SHIPPING Negative đã cao ở TF-IDF. PRICE Negative không có correct Joint predictions ở cả ba families trong các run này; support nhỏ hạn chế kết luận tổng quát. Rare Neutral pairs như PRICE/PACKING vẫn được giữ trong fixed denominator. Heatmap hiển thị đủ 21 pairs với support.\n"
        + figure(4)
        + figure(8)
    )
    text += "\n## 6. Review-Level Error Analysis\n\nMỗi gold-present review–aspect thuộc đúng một outcome: correct, wrong sentiment hoặc missed. Spurious ghi riêng trên gold-absent aspect opportunities. Conditional confusion đọc sentiment bất kể detection; detection-gated confusion có cột MISSED. Hai định nghĩa không được cộng như các event độc lập.\n\n"
    rows = []
    for c in CANDIDATES:
        numbers = []
        for outcome in (
            "CORRECT_ASPECT_CORRECT_SENTIMENT",
            "CORRECT_ASPECT_WRONG_SENTIMENT",
            "MISSED_ASPECT",
            "SPURIOUS_ASPECT",
        ):
            numbers.append(
                value(
                    c + "." + outcome,
                    [
                        sum(
                            r["candidate"] == c and r["seed"] == str(s) and r["outcome"] == outcome
                            for r in d["events"]
                        )
                        for s in SEEDS
                    ],
                    "error_taxonomy.csv",
                    outcome,
                )
            )
        rows.append([c, *numbers])
    text += table(
        ["Candidate", "Correct count", "Wrong sentiment count", "Missed count", "Spurious count"],
        rows,
    )
    text += (
        "\n\nCorrect + wrong + missed bằng 2.969 gold events mỗi run. Spurious denominator là 8.392 gold-absent core aspect opportunities. Wrong sentiment là một event nhưng tạo Joint FP/FN cho hai pairs. Counts không phải số review/khách hàng độc lập. Nguồn: [taxonomy](error_taxonomy.csv), [confusion](confusion_summary.csv), [cross-model](cross_model_error_comparison.csv).\n"
        + figure(6)
    )
    with (OUT / "representative_errors_reviewed.csv").open(encoding="utf8", newline="") as handle:
        examples = list(csv.DictReader(handle))
    rows = []
    for idx, aspect in [(8, "shipping"), (11, "smell"), (12, "texture")]:
        selected = [
            r for r in examples if int(r["source_row_index"]) == idx and r["aspect"] == aspect
        ]
        for c in CANDIDATES:
            if not selected:
                continue
            r = next(r for r in selected if r["candidate"] == c)
            if r["privacy_status"] != "REVIEWED_NO_VISIBLE_IDENTIFIER":
                raise ValueError("unreviewed example")
            rows.append(
                [
                    f"test:{idx}",
                    aspect,
                    r["review_excerpt"],
                    c,
                    r["gold_sentiment"] or "absent",
                    r["conditional_sentiment"],
                    r["outcome"],
                ]
            )
    text += "\n### Reviewed qualitative examples\n\n" + table(
        [
            "Review ID",
            "Aspect",
            "Reviewed excerpt",
            "Model, seed 42",
            "Gold",
            "Conditional output",
            "Outcome",
        ],
        rows,
    )
    text += "\n\nNguồn: [reviewed examples](representative_errors_reviewed.csv). Sampling gồm 50 category slots, 27 unique reviews và 34 review–aspect pairs, với category overlap và seed 42 định trước. Review 11 thể hiện gold/prediction disagreement, chưa chứng minh gold sai. Negation, contrast, slang, emoji và mixed sentiments trong [manual review](test_02c_linguistic_manual.md) là observations, chưa có causal evidence gây lỗi. Mẫu category không phải random sample; privacy review không bảo đảm anonymization tuyệt đối.\n"
    text += "\n## 7. Data Sensitivity\n\n"
    rows = []
    for sub, (n, ceiling) in SUBSETS.items():
        rows.append(
            [sub, n, f"{ceiling}/21"]
            + [
                value(
                    c + "." + sub,
                    [float(d["diagnostics"][c, str(s), sub]["joint_macro_f1"]) for s in SEEDS],
                    "diagnostic_subset_comparison.csv",
                    sub + ".joint_macro_f1",
                )
                for c in CANDIDATES
            ]
        )
    text += table(
        ["Subset", "Reviews", "Support ceiling", "TFIDF-02", "BIGRU-02", "PHOBERT-02"], rows
    )
    text += (
        "\n\nNguồn: [diagnostics](diagnostic_subset_comparison.csv). Ranking giữ nguyên trên các subset có core support; composition khác nhau nên không xem như cùng population. PRICE–Neutral mất support ở nonoverlap L1/L2, ceiling 20/21 nhưng primary denominator vẫn 21. Placeholder-only có 8 review và zero core gold support; Macro-F1 bằng 0 do convention. Chín runs đều có OTHERS TP=8, FP=FN=0 và core FP=0 ở subset này, theo [closure report](test_02c_closure_report.md). NAN origin chưa rõ; L2 collisions không phải semantic duplicate confirmation, L3 chưa đầy đủ; small shifts không chứng minh absence of leakage.\n"
        + figure(7)
    )
    text += """
## 8. Business Discussion

ABSA hỗ trợ đề xuất tập hợp phản hồi theo product và logistics aspects, có human review trước quyết định nghiệp vụ. Negative Recall cao hơn có thể giảm nhãn bị bỏ sót trong test; Precision thấp hơn có thể tăng lượng output cần kiểm duyệt. Chi phí vận hành chưa được đo. Neutral và rare Negative errors giới hạn tự động hóa; Aspect Detection tốt không bảo đảm đúng sentiment. Positive prediction không đo customer satisfaction.

[Business interpretation](business_interpretation.md) mô tả workflow đề xuất; chưa có deployment, ROI, sales, returns hoặc retention evidence. Review/predicted-label frequency không phải complaint prevalence đã hiệu chuẩn ngoài thực tế.

## 9. Threats to Validity

[Threats matrix](threats_to_validity.md) trình bày internal, construct, statistical, data quality, external và privacy risks. Các giới hạn trọng yếu gồm assumed manifest/source equivalence, runtime khác training, shared test, deterministic baseline, rare/zero support, incomplete contamination screening, NAN origin, purposeful category sampling và post-hoc interpretation. Hash checks và arithmetic correctness không xác minh historical training hoặc semantic truth của annotation.

## 10. Main Findings

PhoBERT có mean fixed-21 Joint Macro-F1 cao nhất trong ba selected candidates trên official test, không có significance claim hoặc model reselection. Negative đóng góp lớn vào gap PhoBERT–BiGRU; Neutral contribution nổi bật trong BiGRU–TF-IDF. Hiệu năng theo pair không đồng đều, rare classes hạn chế generalization. Taxonomy tách detection và polarity failures; diagnostics giữ frozen convention. Ứng dụng complaint review có human oversight là đề xuất cần independent deployment validation.

### REPORT-01 / LaTeX handoff

Đây là manuscript Results & Discussion, không thay literature review, methodology đầy đủ hoặc toàn bộ luận văn. Khi chuyển sang LaTeX, giữ table values, captions, source references và metric definitions; đổi headings sang section/subsection và FIG IDs sang label/ref. PNG được include trực tiếp; không sửa template UIT gốc. [Claims ledger](test_02d_claims.json) giữ giá trị đầy đủ, field và nguồn trước khi display rounding. Narrative tiếng Việt, technical labels English.
"""
    summary = "# TEST-02D summary\n\nManuscript, threats matrix, business interpretation và tám figures tổng hợp evidence TEST-01/02B/02C. PhoBERT dẫn mean fixed-21 Macro-F1 trong run quan sát, nhưng minority support và provenance limitations cần được báo cùng scores.\n\n- [Results & Discussion](results_and_discussion.md)\n- [Threats to Validity](threats_to_validity.md)\n- [Business interpretation](business_interpretation.md)\n- [Figures và captions](test_02d_figures_validation.md)\n- [Claims ledger](test_02d_claims.json)\n\nClosure status được xác nhận bằng test_02d_completion.json sau checks. Không có fitting, inference hoặc tuning ở reporting stage.\n"
    return {
        "threats_to_validity.md": threats.encode(),
        "results_and_discussion.md": text.encode(),
        "test_02d_summary.md": summary.encode(),
        "test_02d_claims.json": packed(claims),
    }


def prepare():
    before = snapshot()
    d = load(VAL)
    if d["validation"]["status"] != "PASS":
        raise ValueError("validation evidence blocked")
    outputs = documents(d)
    for name, data in outputs.items():
        if (OUT / name).exists() and (OUT / name).read_bytes() != data:
            raise FileExistsError(name)
    for name, data in outputs.items():
        write_once(OUT / name, data)
    if snapshot() != before:
        raise ValueError("upstream changed")
    print("Manuscript and threats generated; upstream unchanged")


def close(reporting_python):
    import os, subprocess, platform

    before = snapshot()
    prepare()
    generated = {
        name: sha(OUT / name)
        for name in (
            "results_and_discussion.md",
            "threats_to_validity.md",
            "test_02d_summary.md",
            "test_02d_claims.json",
        )
    }
    # Recompute manuscript bytes from evidence, then compare rather than replace.
    prepare()
    if generated != {name: sha(OUT / name) for name in generated}:
        raise ValueError("report nondeterminism")
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/private/tmp/absa-test02d-mpl"
    env["XDG_CACHE_HOME"] = "/private/tmp/absa-test02d-cache"
    primary = sys.executable
    commands = [
        [primary, "-B", "tests/test_protocol.py"],
        [primary, "-B", "tests/test_preflight.py"],
        [primary, "-B", "tests/test_test02b.py"],
        [
            primary,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_test02c*.py",
            "-v",
        ],
        [
            reporting_python,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_test02d*.py",
            "-v",
        ],
        [primary, "-B", "-m", "src.analysis.test02d", "--check-stale"],
        [primary, "-B", "-m", "src.analysis.test02c.closure"],
    ]
    results = []
    for cmd in commands:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        results.append({"command": cmd, "exit_code": proc.returncode, "output": output})
        print("CHECK", cmd, "exit", proc.returncode, flush=True)
        if proc.returncode:
            raise RuntimeError(output)
    # Re-run all eight renderers twice; existing outputs must compare byte-identical.
    for attempt in (1, 2):
        cmd = [
            reporting_python,
            "-B",
            "-m",
            "src.analysis.test02d",
            "--validation-aggregate",
            "../absa-run/validation_aggregate_assumed_equivalence.json",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        results.append(
            {
                "command": cmd,
                "attempt": attempt,
                "exit_code": proc.returncode,
                "output": proc.stdout + proc.stderr,
            }
        )
        print("Figure regeneration", attempt, "exit", proc.returncode, flush=True)
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
    if before != snapshot():
        raise RuntimeError("existing evidence changed")
    # One final computation reads sealed source values and validates report identity.
    data = load(VAL)
    for name, raw in documents(data).items():
        if (OUT / name).read_bytes() != raw:
            raise ValueError("report no longer matches evidence")
    business = OUT / "business_interpretation.md"
    business_text = business.read_text()
    for c in CANDIDATES:
        for sentiment in ("negative", "neutral"):
            expected = []
            for metric in ("precision", "recall", "f1"):
                vals = [
                    float(data["sentiment"][c, str(seed), sentiment][metric]) for seed in SEEDS
                ]
                expected.append(
                    f"{statistics.mean(vals)*100:.2f} ± {statistics.stdev(vals)*100:.2f}".replace(
                        ".", ","
                    )
                )
            if "| " + c + " | " + " | ".join(expected) + " |" not in business_text:
                raise ValueError("business metric mismatch " + c + sentiment)
    manifest = json.loads((OUT / "figure_manifest.json").read_text())
    if stale(manifest):
        raise ValueError("final stale figures")
    code_paths = [
        ROOT / "src/analysis/test02d_reporting.py",
        ROOT / "tests/test_test02d_reporting.py",
    ]
    integrity = {
        "status": "PASS",
        "input_sha256": before,
        "upstream_unchanged": True,
        "output_sha256": generated,
        "figure_manifest_sha256": sha(OUT / "figure_manifest.json"),
        "code_sha256": {str(p): sha(p) for p in code_paths},
        "checks": results,
        "report_reproducibility": "byte-identical regeneration",
        "figure_reproducibility": "two identical renders in the recorded Python/Matplotlib environment",
        "runtime": platform.python_version(),
        "figure_environment": manifest["figures"][0]["environment"],
        "scope": "TEST-02D reporting only; REPORT-01 LaTeX conversion excluded",
        "limitations": [
            "ASSUMED_MANIFEST_EQUIVALENCE: six neural links remain historically unverified.",
            "Historical source fingerprint and original training runtime not independently reproduced.",
            "Privacy review cannot guarantee anonymization; no causal/statistical significance claims.",
        ],
        "fitting_inference_tuning_performed": False,
    }
    write_once(OUT / "test_02d_integrity.json", packed(integrity))
    report = """# TEST-02D Closure Report

Status: TEST-02D COMPLETE — completion receipt is test_02d_completion.json.

## Delivered

- Eight figures with captions, hashes and current figure manifest.
- Evidence-grounded business interpretation.
- Results & Discussion manuscript and full-precision claims ledger.
- Threats to Validity matrix and research summary.

## Verification

Existing protocol, preflight, TEST-02B and TEST-02C tests plus TEST-02D figure/report tests ran successfully. Exact commands, process output and return codes are retained in test_02d_integrity.json. Mean/sample SD, secondary metrics, fixed-21 decomposition, reviewed excerpts and local evidence references are checked. All input hashes remain unchanged.

Manuscript regeneration was byte-identical. All eight figures were regenerated twice in the reporting environment; existing PNGs and manifests remained byte-identical. Stale check returned CURRENT. This does not require cross-platform byte-identical rendering.

## Research limits and handoff

Historical neural manifest/source equivalence remains assumed, not verified. Three seeds share one test set. Low support, incomplete overlap screening, unresolved NAN origin, annotation ambiguity and manual privacy limitations remain in threats_to_validity.md. No training, inference, model reselection, checkpoint/threshold changes, commit or push occurred.

The Markdown manuscript and figure captions are ready for REPORT-01 conversion. The UIT template is unchanged; literature review, full thesis methodology/front matter and LaTeX compilation belong to REPORT-01. Earlier PARTIAL receipts describe earlier stages and are preserved.
"""
    write_once(OUT / "test_02d_closure_report.md", report.encode())
    artifact_names = sorted(OUTPUTS - {"test_02d_completion.json"})
    completion = {
        "status": "TEST-02D COMPLETE",
        "artifacts_sha256": {name: sha(OUT / name) for name in artifact_names},
        "figure_manifest_sha256": sha(OUT / "figure_manifest.json"),
        "all_checks_passed": True,
        "runs": 9,
        "figures": 8,
        "provenance": "ASSUMED_MANIFEST_EQUIVALENCE; historical training source/runtime not fully verified",
        "upstream_evidence_unchanged": True,
        "training_inference_or_tuning_performed": False,
        "report01_latex_conversion": "NOT_EXECUTED",
    }
    write_once(OUT / "test_02d_completion.json", packed(completion))
    print("TEST-02D COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close", action="store_true")
    parser.add_argument(
        "--reporting-python", default="/private/tmp/absa-test02d-reporting/bin/python"
    )
    args = parser.parse_args()
    if args.close:
        close(args.reporting_python)
    else:
        prepare()
