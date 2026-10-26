# Data & Evaluation Protocol Freeze v1

Dataset: **original_shopee_absa_v1**. Protocol: **absa_data_evaluation_v1**.

Mục đích là cố định dữ liệu và evaluator trước khi so TF-IDF, BiGRU/Conv1D và PhoBERT.
Freeze này không phải EDA mới, cleaned dataset hoặc checkpoint huấn luyện.
Các quyết định mặc định bổ sung ở đây là lựa chọn triển khai: 3 seeds `42/2026/3407`, mean-validation selection,
tie-break và một global aspect threshold cho mỗi model. Chúng được ghi rõ trong YAML trước thí nghiệm.

## 1. Nguồn dữ liệu và phạm vi freeze

| Split | File | Reviews |
|---|---|---:|
| Full, mirror corpus | `data/full_data.csv` | 16.227 |
| Train | `data/data_train.csv` | 12.981 |
| Validation (`val`) | `data/data_val.csv` | 1.623 |
| Test | `data/data_test.csv` | 1.623 |

SHA-256 từng file nằm trong `evaluation_protocol_v1.yaml`, đối chiếu với EDA-04, không tự lấy fingerprint mới để chấp nhận drift.
Full là multiset union của ba split, giữ multiplicity. Không nối full vào train/val/test.
Original benchmark giữ nguyên row count, nội dung, nhãn, thứ tự và split. Không sửa 16 all-blank reviews.

File thực tế đang có là `src/eda.ipynb` và `src/eda_nlp.ipynb`; không có `eda(1).ipynb` trong checkout.
Freeze dùng chúng như hai bằng chứng EDA được nêu trong yêu cầu, giữ snapshot hashes của chúng trong lock và không sửa notebook.
Notebook có thể được bổ sung sau này; hashes này chỉ là provenance của thời điểm freeze, không phải input cho model.

## 2. Raw text, audit normalization và model input

`raw_text` là giá trị field `data` sau CSV parsing, không phải raw bytes của cả record.
Đọc UTF-8 với CSV quoting/newlines đúng; dùng `csv` giữ mọi string, kể cả `NAN`, `NA` và blank. Nếu dùng pandas phải có `keep_default_na=False`.
Giữ nguyên Unicode, case, dấu câu, emoji, xuống dòng và whitespace trong field. Raw CSV records cũng được hash/reparse riêng.

### Audit levels

| Level | Định nghĩa chính xác | Vai trò |
|---|---|---|
| L0 | Raw parsed text identity, giống từng Unicode code point | Exact text duplicates |
| L1 | Python `lower()` → `strip()` → `re.sub(r'\s+', ' ', text)` | Primary contamination report |
| L2 | Unicode NFC → lower → `re.sub(r'[^\w\s]', ' ', text)` → `re.sub(r'[\s_]+', ' ', text).strip()` | Aggressive candidate sensitivity |
| L3 | Bounded near-duplicate screen trong `eda_nlp.ipynb` | **Incomplete candidate screening**, chưa exhaustive |

L2 khớp code notebook hiện tại: regex bỏ cả emoji/ký hiệu ngoài `\w/\s`, không chỉ ASCII punctuation; không bỏ dấu tiếng Việt và vẫn giữ số.
Các phép biến đổi không tạo phân cấp tập hợp tuyệt đối: NFC có thể merge nhóm trong khi punctuation/mark replacement làm thay đổi nhóm khác.
Không diễn giải chênh lệch duplicate count là số review gốc bị sao chép thêm.

Full duplicate-text extras: **L0=257, L1=274, L2=338**. Extra = số rows trừ số keys, gồm empty key nếu phép biến đổi tạo ra nó.
L2 tạo **31 empty-key rows** trên full. Nhóm này được tính trong duplicate census để khớp notebook,
nhưng key rỗng **không** được dùng để xác nhận cross-split overlap hoặc loại test row khỏi nonoverlap subset.
Giữ `empty_key_l2=True` để tránh ngầm coi dấu câu/emoji khác nhau là cùng nội dung gốc.

`has_label_conflict_l*` nghĩa là một text key có hơn một vector 8 nhãn, kể cả khác blank/present.
`has_polarity_conflict_l*` chỉ khi cùng text key và cùng aspect có nhiều **nonblank** sentiments.
Trên corpus hiện tại, L0/L1/L2 có label-vector conflicts nhưng không có nonblank polarity conflicts.
Điều này nhấn mạnh vấn đề annotation presence/absence; chưa xác nhận nhãn nào sai.

L3 cũ: tối đa 5 rare contiguous trigrams/text, postings ≤100, cap 30.000 candidate pairs,
length ratio ≥0,8, SequenceMatcher ≥0,90, loại normalized exact matches. Có retrieval-order bias nếu đạt cap,
có thể bỏ sót short text/paraphrase. Freeze không chạy lại hoặc gọi kết quả đó là prevalence.

### Audit manifest

`data_audit_manifest.csv` có **16.227 rows canonical**, mỗi official split row đúng một lần, theo train → val → test.
Không nhân đôi corpus bằng cách thêm full thành 16.227 mẫu mới.

- `source_split`, `source_row_index` (0-based parsed record), source file/SHA, physical CSV line start/end và raw-record SHA-256.
- `full_source_row_index`, full physical lines và raw-record SHA: bijection với toàn bộ full file.
  Mapping xử lý identical occurrences bằng queue theo thứ tự. Đây là occurrence alignment để audit, **không** phục hồi source review ID không tồn tại.
- `raw_text`, 8 raw labels, raw L0 key, normalized L1/L2 keys, stable group IDs = level + SHA-256 của key UTF-8.
- Global group size tính trên official split corpus (không cộng full mirror), same-split group size, label/polarity conflict flags từng level.
- `overlap_{train,val,test}_{l0,l1,l2}` nghĩa là key nonempty có mặt ở **split khác** tương ứng; same-split duplicate không bị gọi là cross-split overlap.
- Aliases `duplicate_group_id`, `has_label_conflict`, `overlap_train`, `overlap_validation` dùng **L1**.
- Placeholder/empty-key flags, audit status, L3 incomplete status và diagnostic subset flags.

Manifest chứa review text để truy nguyên; các trường audit, source IDs và labels không được đưa vào model features.

### Model preprocessing (tách khỏi audit)

L0/L1/L2 keys **không phải model input**. Model input bắt đầu từ raw text, rồi transform riêng có version:

| Family | Baseline preparation | Gate trước huấn luyện |
|---|---|---|
| TF-IDF + LR/SVM | NFC + whitespace cleanup; word/bigram và char n-grams; case settings ghi trong experiment config | Fit vocabulary/IDF trên train; ghi vectorizer settings, feature transform/version |
| BiGRU/Conv1D | NFC + whitespace cleanup; tokenizer/vocabulary riêng, không blanket-remove negation/punctuation | Fit vocabulary trên train, freeze tokenization, embedding, truncation và architecture |
| PhoBERT | NFC + whitespace cleanup, **Vietnamese word segmentation trước PhoBERT tokenizer** | Pin segmenter/model/tokenizer revisions; audit P95/P99, special tokens, truncation 128/256 theo model limit |

VinAI hướng dẫn input word-segmented và khuyến nghị RDRSegmenter/VnCoreNLP cho raw Vietnamese input.
Đây là chuẩn bị model còn **pending**, không có tokenizer/model download hoặc token-length claim trong freeze này.
Nguồn: [PhoBERT official README](https://github.com/VinAIResearch/PhoBERT).

## 3. Label semantics và target order

Label normalization v1 là **strict identity**: chỉ `positive`, `neutral`, `negative`, `''` hợp lệ. Không tự lower/trim nhãn sai rồi chấp nhận.
Blank = aspect chưa được gán nhãn trong benchmark, không phải Neutral và không chứng minh aspect không được nhắc đến thực tế.

- Aspect detection: 8 binary labels theo thứ tự YAML, gồm OTHERS.
- Core sentiment: 7 aspects, mỗi aspect 3 classes theo `positive/neutral/negative`.
- Joint: 21 pairs, **aspect-major, sentiment-minor**, thứ tự explicit trong YAML.
- OTHERS: spam detection riêng; không có `others:neutral` trong core joint labels.
- Có 4 train reviews mang OTHERS cùng core: không mutually exclusive và không hard discard review vì OTHERS.
- Per-core-aspect model output có đúng một conditional sentiment. Joint output được gate bằng predicted aspect,
  nên bỏ sót aspect tạo FN; nhầm aspect tạo FP/FN. Conditional diagnostic không gate bằng predicted presence.

## 4. NAN audit và hai tracks

Raw source có **105 exact `NAN`**, train/val/test **84/13/8**, tất cả chỉ `OTHERS=neutral`; 105 row vectors giống nhau.
Không thấy variant khác trong lexicon frozen `nan/null/none/n/a/na/<na>/nil` sau casefold+trim.
Chuỗi này có sẵn trong CSV, không phải do pandas conversion trong audit. **Nguồn gốc chưa xác minh.**

Paper có “nan” trong output encoding cho absent aspect; điều đó không giải thích vì sao review text là `NAN`.
Không gọi 105 rows là confirmed collection error hoặc xoá khỏi benchmark.

Sensitivity full nếu loại exact NAN: reviews **16.227 → 16.122**, duplicate-row extras **251 → 147**,
OTHERS **2.872 → 2.767**. NAN đóng góp **104/251 ≈41,43%** duplicate-row extras, dù chỉ ~0,647% corpus.
Whitespace P50/P95 vẫn **18/50**; multi-aspect % tăng do mẫu số giảm, không phải gán thêm aspects.
Bảng đủ splits/length/duplicate statistics ở `reports/protocol_freeze/NAN_normalization_audit_v1.md`.

- **Track A chính:** original benchmark, tất cả reviews và splits gốc.
- **Track B evaluation filter:** cùng checkpoint, original test và without-placeholder/placeholder-only diagnostic subsets.
- **Track B training ablation:** run riêng, chỉ bỏ placeholder candidates trong train, giữ evaluation protocol/subsets và validation policy đã khai báo.
  Không đổi tên run đó thành replication của benchmark gốc. Không chọn filter/loss bằng điểm test.

Việc bỏ OTHERS column khỏi core sentiment counts khác với bỏ review. Placeholder test chỉ có OTHERS nên không có core gold positives;
vẫn đếm core false positives nếu model dự đoán nhầm. Chưa có predictions để đo score inflation hoặc hiệu quả bỏ placeholder.

## 5. Evaluation & reporting

Machine-readable source: `evaluation_protocol_v1.yaml` dùng JSON syntax hợp lệ trong YAML 1.2 để không cần dependency parser riêng.
Implementation: `src/absa_protocol.py`; CLI: `src/evaluate_predictions.py`.

### Metric rules

| Result | Unit / averaging |
|---|---|
| Primary joint Macro-F1 | Review × **21 fixed pairs**, binary PRF theo pair, mean tất cả 21; zero-denominator F1=0 |
| Joint Micro/Weighted-F1, P/R | Pool TP/FP/FN cho micro; gold support weights cho weighted; macro P/R undefined contributes 0 |
| Aspect detection | 8 fixed labels Macro/Micro/Weighted PRF, per-aspect support và confusion counts |
| Conditional sentiment | Chỉ gold-present examples theo aspect; 3 fixed classes/aspect; primary conditional summary = mean 7 aspect macro-F1, empty aspect contributes 0 và có unevaluable list |
| Joint per-aspect | 3 fixed pair labels/aspect, vẫn tính detection mistakes |
| Joint per-sentiment / complaints | Pool TP/FP/FN của 7 pairs cùng sentiment; Negative P/R/F1, thêm từng pair |
| OTHERS | Binary detection PRF + support; ngoài core sentiment |

F1 lớp absent cả gold và prediction = **0 theo convention**, không phải bằng chứng model thất bại trên lớp chưa được kiểm tra.
Per-class recall nếu gold support=0 = **JSON null / N/A**. Precision nếu không có predicted positives cũng null.
Aggregate macro P/R dùng 0 cho undefined contribution; weighted P/R nếu tổng support=0 dùng 0 theo convention, giữ zero-support list.
Secondary `observed_support_macro_f1` chỉ mean labels có gold support>0; không lẫn với primary và vẫn phải report FP ở lớp absent.
Subset 0 review: `status=empty_subset`, aggregate scores=null, không gọi score=0.
Reference semantics: [scikit-learn F1 docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
Evaluator dùng công thức TP/FP/FN stdlib; unit tests tính tay không phụ thuộc sklearn version.

### Dataset subsets, membership frozen trước predictions

| Subset | Reviews | PRICE-Neutral support |
|---|---:|---:|
| Original test | 1.623 | 1 |
| `test_nonoverlap_l1` | 1.592 | 0 |
| `test_nonoverlap_l2` | 1.585 | 0 |
| `test_without_placeholder` | 1.615 | 1 |
| `test_placeholder_only` | 8 | 0 |

Overlap reference = train **hoặc** validation, nonempty key; same test duplicates không tự loại.
L1 primary contamination: 31/1.623 test rows. L2 sensitivity loại 38 rows theo union train/val (pairwise counts không được cộng trực tiếp).
PRICE-Neutral test row 720 trùng train row 12120 cả text+labels; không giả vờ subset nonoverlap còn đánh giá được lớp này.
L2 nonoverlap chưa phải confirmed clean benchmark; near-duplicate và product/user/time leakage chưa được loại trừ.

### Prediction interface

JSONL, một object cho **mỗi row của cả split** (`val` hoặc `test`), bắt buộc có key ID và outputs:

```json
{"source_split":"test","source_row_index":0,"aspects":["packing"],"conditional_sentiments":{"stayingpower":"positive","texture":"positive","smell":"positive","price":"positive","colour":"positive","shipping":"positive","packing":"positive"}}
```

Đây là ví dụ schema, không phải prediction measurement. Cần conditional output cho **tất cả 7 aspects** để đánh giá gold-aspect sentiment,
kể cả khi aspect detector bỏ sót. Có thể dự đoán OTHERS đồng thời core aspects; unknown/duplicate IDs hoặc labels, thiếu row/output sẽ fail.
CLI align theo split+row index, không tin thứ tự JSONL; cùng predictions/protocol sinh cùng JSON metrics.

```bash
python3 src/evaluate_predictions.py --split val --predictions /path/to/predictions.jsonl --output /path/to/metrics.json
# Sau khi checkpoint, code, thresholds và run config đã freeze:
python3 src/evaluate_predictions.py --split test --predictions /path/to/test.jsonl --output /path/to/test_metrics.json
```

Evaluator chạy preflight trước scoring và tự sinh đủ diagnostic reports từ một file predictions test.
Không tự tạo fake model metrics, không overwrite report. Lưu prediction SHA, config SHA và freeze-lock SHA trong report.

## 6. Model selection và experiment gate

1. Train-only fit: vocabulary/IDF, weights, sampling statistics, tokenizer length statistics. Không đọc test để chọn method.
2. Đăng ký candidate configs/grids và resource budget trước fit cho từng family. Phải ghi chính xác transform/tokenizer/segmentation revisions.
3. Chọn config theo **mean validation joint Macro-F1 trên 3 seeds 42/2026/3407**; tie: mean joint Micro-F1, rồi candidate ID thấp hơn đã đăng ký.
4. Một global aspect threshold/model, chọn trong grid đã đăng ký bằng val primary; conditional argmax ties theo sentiment order.
   Không per-rare-class threshold tuning khi thiếu support. Nếu model có decision score thay probability, ghi domain và grid tương ứng.
5. PRICE-Negative có val support=0: không tuyên bố validation đã tối ưu recall lớp đó. Train-only grouped CV **disabled trong v1**;
   muốn thêm thì preregister extension trước khi chạy, không tự bật dựa trên test.
6. Freeze checkpoints, preprocessing, threshold, code+dirty patch hash, lock hash, package/hardware versions và all seeds trước final test.
7. Một final evaluation round cho các models/seeds/checkpoints đã đăng ký, kèm các subsets cố định. Report mọi seed, không chọn best test seed.

TF-IDF là traditional baseline; BiGRU/Conv1D là recurrent DL comparator; PhoBERT là contextual DL comparator.
Preserve split/target/evaluator semantics trong cả ba. Hyperparameter grids cụ thể, tokenizer and checkpoint revisions, sequence length,
GPU budgets và training scripts chưa có trong vòng freeze này; run registration là điều kiện trước model fitting.
Giữ benchmark không đồng nghĩa metric ở đây tự động tái lập metric paper: muốn numerical replication cần version protocol paper riêng rõ averaging/decoding.

## 7. Reproduction và acceptance

Chạy từ repository root, Python **3.10+**, không cần pip packages cho preflight/evaluator/tests.
Lock ghi runtime tạo freeze: Python **3.15.1**, Unicode database **17.0.0** ở máy hiện tại.
Preflight yêu cầu cùng Unicode DB version để lower/regex/NFC semantics không âm thầm drift; dùng runtime tương ứng hoặc version protocol nếu đổi.

```bash
python3 src/preflight_validation.py --check
python3 -m unittest discover -s tests -v
```

`--check` chỉ đọc: xác minh source fingerprints, full/split multisets, labels, manifest bijection/provenance,
recompute byte-for-byte manifest/summary/report, subset memberships, deterministic statistics và sealed artifact hashes.
Tests riêng kiểm tra metric edge cases, wrong aspect, OTHERS+core, normalization, ID alignment và tamper rejection.

Artifacts đã được tạo bằng `--build` và seal bằng `--seal`; **không chạy lại để chấp nhận drift**.
Build/seal từ chối overwrite khác nội dung. Thay đổi có chủ đích phải thành v2, không refresh v1 hash cho qua check.
Lock là integrity snapshot local, không phải chữ ký chống sửa đổi malicious. Không commit/push trong vòng làm việc này.

### Pre-training readiness: data & evaluator scope

- [x] Cố định dataset và xác nhận SHA-256.
- [x] Định nghĩa L0/L1/L2, counts và pairwise/union overlaps; L3 gắn incomplete.
- [x] NAN raw-record audit, label/split census và sensitivity table; **origin vẫn unresolved được công khai**.
- [x] Blank, OTHERS/core và output order 8/7/21.
- [x] Fixed-label metrics, zero-support và diagnostic subset memberships.
- [x] Train/validation-only selection, 3 seeds, tie-break và no-test-retuning rules.
- [x] Preflight và metric tests được chạy; kết quả lần thực thi lưu trong `reports/protocol_freeze/verification_v1.json`.

**Chưa phải model-ready:** tokenizer-only audit PhoBERT (segment → subword → P95/P99 → 128/256 truncation → special tokens)
và model-specific run registration còn pending. Không dùng checklist này để tuyên bố đã sẵn sàng train PhoBERT hoặc đã chọn max_length.

## 8. Giới hạn và provenance nguồn

NAN origin chưa được tác giả xác minh; không tìm thấy giải thích trong README, notebook và phần paper đã kiểm tra.
Label conflicts chủ yếu là presence/absence; qualitative flags chưa adjudicate. Exact/normalized overlaps là census,
L2 equality không xác nhận same-origin copy, L3 không exhaustive. Test đã được xem trong descriptive EDA;
không được tiếp tục lựa chọn method từ test. Chưa đo được score inflation hoặc lợi thế DL.

Tài liệu tác giả: [PACLIC 2022 paper §3–4](https://aclanthology.org/2022.paclic-1.84.pdf).
Paper bàn output “nan” là absent aspect; điều đó khác literal review `NAN`.
Nguồn kỹ thuật đã đối chiếu: [VinAI PhoBERT README](https://github.com/VinAIResearch/PhoBERT),
[scikit-learn F1 docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
Model/tokenizer package version chưa cài không được coi là pinned chỉ vì trang docs đang có một version number.
