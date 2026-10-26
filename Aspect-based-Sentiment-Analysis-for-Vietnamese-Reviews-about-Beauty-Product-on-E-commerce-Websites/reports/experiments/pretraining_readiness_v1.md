# Báo cáo thực nghiệm — kiểm tra trước training

Phạm vi: reproducibility gate, diễn giải metric và modeling preflight / đăng ký thí nghiệm.
Chưa huấn luyện hoặc đánh giá model. Những lựa chọn hyperparameters bên dưới là baseline defaults đăng ký trước,
không phải cấu hình đã chứng minh tối ưu. Protocol dữ liệu/evaluator **giữ nguyên v1**.

## 1. Reproducibility gate

- Runtime thực tế: **Python 3.15.1 / Unicode 17.0.0**.
- Full `preflight_validation.py --check`: **PASS**. Cùng source fingerprints, multiset, manifest, normalization và diagnostic subsets đã freeze.
- 16 unit tests: **PASS**, gồm 13 tests gốc và 3 tests mới cho ceiling interpretation / token length / model text preparation.
- Lock v1 trước/sau vẫn `16e9b76b6a71530d08704cadc2d1ddeddf54ca57615c35f8e650ccc57ac86ee8`.
- Không chạy `--build`/`--seal` để refresh checksum v1; không thay frozen source/evaluator/metric formula.

Chi tiết command/stdout/stderr: [reproducibility_gate_v1.json](../modeling_preflight/reproducibility_gate_v1.json).

## 2. Đoạn diễn giải bắt buộc trong báo cáo kết quả

Primary joint Macro-F1 của nghiên cứu lấy trung bình **21 core aspect–sentiment labels cố định** với zero-division F1=0.
Nếu một tập có m labels có gold support>0, support-only upper bound là **m/21**. Vì thế, một model/oracle dự đoán
đúng toàn bộ observed targets vẫn không nhất thiết đạt 100% primary Macro-F1 trên một subset thiếu nhãn.

| Tập đánh giá | Supported labels | Upper bound theo support | Nhãn không được kiểm tra |
|---|---:|---:|---|
| Official validation | 20/21 | 95,2381% | PRICE-Negative |
| Original test | 21/21 | 100% | — |
| Test non-overlap L1 | 20/21 | 95,2381% | PRICE-Neutral |
| Test non-overlap L2 | 20/21 | 95,2381% | PRICE-Neutral |
| Test without placeholder | 21/21 | 100% | — |
| Test placeholder-only | 0/21 | 0% | Toàn bộ core pairs; OTHERS vẫn có thể đánh giá |

Đây là ceiling do convention và support, **không phải observed performance, không phải mức model được dự báo sẽ đạt**.
Absent-label F1=0 không chứng minh model thất bại trên một class chưa được kiểm tra. Recall ở gold support=0 ghi N/A;
false positives vẫn phải báo. Conditional summary fixed7×fixed3 có ceiling theo support tương ứng; per-aspect ceiling là số supported sentiments /3.

Giữ primary fixed21, report observed-support macro riêng, cùng TP/FP/FN/support và Negative P/R.
Không chia primary score cho ceiling, không ngầm bỏ absent classes, không so raw macro giữa subsets như chúng có cùng composition.
Nếu đổi công thức, denominator hoặc zero_division thì phải có **evaluation protocol v2**. Việc thêm diễn giải này giữ v1.

Chi tiết và số liệu machine-readable: [reporting_guidance_v1.md](reporting_guidance_v1.md),
[macro_f1_ceiling_v1.json](macro_f1_ceiling_v1.json).

## 3. PhoBERT tokenizer / word segmentation đã chạy thực tế

Pipeline local: raw train text → NFC + whitespace cleanup (giữ case/dấu câu/emoji/negation) →
**VnCoreNLP 1.2 / RDRSegmenter wseg-only** → join sentences bằng một space → **PhobertTokenizer slow** → BOS/EOS.
Từng review được segment riêng bằng Java API, không nối các review thành một câu hoặc chèn marker vào văn bản.

- PhoBERT: `vinai/phobert-base`, revision `01daacda68afe13d83023d16ec647239e344a1e6`.
- VnCoreNLP revision: `62bbc58fe5d113c898eae112656be97dcf50b3a0`, jar + 2 wordsegmenter models được hash/pin.
- Java portable: Temurin **17.0.21+9**, archive kiểm tra với SHA-256 từ metadata release.
- Modeling runtime riêng: **Python 3.13.9 / Unicode 15.1.0**, `transformers==4.57.1`; mọi dependency tokenizer thực tế có trong environment snapshot.
- Runtime modeling riêng không thay thế gate Python 3.15.1. Script luôn gọi lại gate bằng interpreter khóa trước audit.
- Audit trên **toàn bộ 12.981 train reviews**, giữ cả 84 NAN. Không dùng val/test token lengths để chốt max_length.

| Thống kê subword, có special tokens | Giá trị |
|---|---:|
| P50 | 26 |
| P95 | 64 |
| P99 | 92 |
| Max | 301 |
| Truncation @128 | 41 / 12.981 = 0,31585% |
| Truncation @256 | 2 / 12.981 = 0,01541% |

Rule chọn trước kết quả model: dùng 128 nếu train P99≤128 và truncation@128≤1%, ngược lại 256.
Kết quả: **đăng ký PhoBERT max_length=128**. Hai special tokens BOS=0, EOS=2; PAD=1, UNK=3, MASK=64000.
Đã kiểm tra số special tokens, EOS sau truncation, alignment đủ records và smoke examples cho segmentation/negation/emoji/NAN/punctuation.
Các cặp train PRICE-Negative, PRICE-Neutral và PACKING-Neutral không có review nào bị truncation ở 128 trong audit này.

### Coverage limitations cần báo cáo

15.844 / 371.706 bare subwords là UNK (**4,2625%**); **2.032/12.981 reviews** chứa ít nhất một UNK.
Ví dụ `👍🏻` được giữ trong segmented text nhưng trở thành hai UNK. Giữ emoji trong raw input không đồng nghĩa tokenizer giữ đầy đủ sentiment signal.
RDRSegmenter cũng có thể join lạ trong ngữ cảnh review: smoke example có `Nhưng_son`, `biết_son`.
Đây là limitations của pipeline đã chọn, chưa có manual segmentation gold để định lượng lỗi.
Không sửa riêng các ví dụ đó rồi coi là pipeline tổng quát; thay segmentation/emoji mapping là ablation cần đăng ký riêng trước fit.

Không có review train nào ra segmented text rỗng. Không tải pretrained model weights hoặc chạy backward/optimizer.
Chi tiết: [phobert_tokenizer_audit_v1.json](../modeling_preflight/phobert_tokenizer_audit_v1.json).
Cache từng train row có source index/raw-text hash, segmented text và full untruncated token IDs:
[train_tokenization_v1.jsonl](../modeling_preflight/train_tokenization_v1.jsonl).

## 4. Đăng ký trước các thí nghiệm

Machine-readable: [experiment_registration_v1.json](../../protocols/experiments/experiment_registration_v1.json).
Registration có lock riêng, chỉ tham chiếu original freeze-lock; không reseal/đổi checksum v1.

| Family | Candidates | Key defaults | Budget / seed |
|---|---|---|---|
| TF-IDF + Logistic Regression | Word (1,2); word (1,2) + char (3,5) | Train-only vocab/IDF, C=1, liblinear OVR, giữ token `k`; 8 aspect và 7 sentiment classifiers | 1.000 solver iterations, tối đa 30 phút |
| BiGRU / BiGRU+Conv1D | BiGRU; BiGRU + parallel Conv1D | RDR segmentation, vocabulary train-only 30k, embedding128, hidden128/direction, kernel3/128 filters, max_length128, Adam lr1e-3 | ≤20 epochs, patience3, ≤60 phút |
| PhoBERT | LR1e-5; LR2e-5 | Pinned model/segmenter/tokenizer, max_length128, BOS representation, 8+7×3 heads, AdamW, effective batch32 | ≤5 epochs, patience2, ≤120 phút |

**2 candidates × 3 families × 3 seeds = tối đa 18 main fits**. Seeds: **42, 2026, 3407**.
Neural loss: aspect BCE mean + conditional CE masked theo gold-present core aspects; không class weights/focal ở main baseline.
Threshold grid chung `[0.3,0.4,0.5,0.6,0.7]`, một global aspect threshold/checkpoint cho cả 8 labels; không per-rare-class tuning.
Chọn configuration bằng mean official-validation primary qua cả 3 seeds, tie theo micro rồi preregistered candidate ID;
checkpoint/threshold tie-break được ghi rõ. Không chọn best test seed hoặc retune bằng diagnostic subsets.

Torch/sklearn training dependency graph đã resolve và hash-lock ở
[modeling_requirements_v1.lock](../../protocols/experiments/modeling_requirements_v1.lock), gồm `torch==2.9.0`, `scikit-learn==1.7.2`.
Chỉ tokenizer environment đã được cài và chạy ở vòng này. Full training stack / CUDA chưa cài hoặc đo tại đây.
Neural fitting được đăng ký cho single CUDA GPU ≥16 GiB, FP32, AMP/TF32 off; package/driver/hardware và deterministic settings phải được kiểm tra ở launch.
Budget khác nhau giữa families được khai báo trước; không diễn giải kết quả tương lai là so sánh dưới equal wall-clock compute.

## 5. Cách chạy lại và gate còn lại

Từ repository root:

```bash
# 1. Exact-runtime reproducibility check; không build/seal lại v1
python3 src/reproducibility_gate.py --output reports/modeling_preflight/reproducibility_gate_v1.json

# 2. Verify đăng ký / artifact hashes
python3 src/register_experiments.py --verify

# 3. Khôi phục assets từ URL/revision/SHA đã pin nếu thư mục cache tạm không còn
python3 src/restore_model_assets.py --assets /private/tmp/absa-model-assets-v1

# 4. Tokenizer-only rerun trong Python 3.13.9 environment có deps của tokenizer_environment_v1.txt
/path/to/modeling-python src/modeling_preflight.py \
  --assets /private/tmp/absa-model-assets-v1 \
  --gate-python /path/to/python3.15.1
```

Thư mục cache assets/environment cục bộ ở `/private/tmp`, có thể bị xoá bởi hệ điều hành.
Receipts/revisions/package pins, script restore và hash-locked requirements được lưu trong repo để tái tạo; không cần tải model weights cho tokenizer audit.

**Trước khi fitting thực sự:** triển khai/kiểm tra trainer dùng đúng registered decoding/loss/evaluator,
kiểm tra training dependencies và target GPU, lưu run manifest với exact trainer code/dirty hash, hardware, config/lock hashes.
Sau selection mới freeze selected checkpoints/thresholds rồi chạy một final test round.
Vòng này hoàn thành data gate, reporting interpretation và tokenizer/config registration; chưa xác nhận GPU training runtime hoặc model performance.
