# NAN & normalization audit v1

Đây là sensitivity analysis; benchmark `original_shopee_absa_v1` giữ nguyên tất cả dòng/nhãn/splits.

## NAN: xác nhận và chưa xác nhận

- Raw CSV có đúng 105 field text `NAN`: train 84, val 13, test 8. Tất cả chỉ có OTHERS:Neutral.
- Mỗi record gốc đã được reparse; manifest có source record index, physical line start/end và raw-record SHA-256.
- 105 row vectors giống nhau hoàn toàn: đóng góp 104 duplicate-row-extra trên full. Giữ `NAN` là string; không ép thành missing.
- Không thấy biến thể khác trong lexicon casefold+trim: NAN/nan/NaN, NULL, None, N/A, NA, <NA>, nil.
- Chưa xác minh nguồn gốc: paper dùng “nan” để nói aspect absent, không giải thích literal review `NAN`. Không đánh đồng hai điều này.
- Đã đọc paper §3–4 và README/notebook hiện có; chưa tìm được giải thích tác giả về việc tạo chuỗi này. Đây là giới hạn tìm kiếm, không chứng minh tài liệu đó không tồn tại.

## Original vs excluding literal NAN (đếm riêng, không cộng full với splits)

| Split | Rows original → excl | Duplicate row extra | OTHERS count | Multi-aspect % | Word P50/P95 | Char P50/P95 |
|---|---:|---:|---:|---:|---|---|
| full | 16227 → 16122 | 251 → 147 | 2872 → 2767 | 59.444 → 59.831 | 18/50 → 18/50 | 81/223 → 81/224 |
| train | 12981 → 12897 | 180 → 97 | 2291 → 2207 | 59.595 → 59.983 | 18/50 → 18/50 | 81/222 → 81/222 |
| val | 1623 → 1610 | 18 → 6 | 270 → 257 | 58.534 → 59.006 | 18/48 → 18/48 | 80/213.7 → 81/215.1 |
| test | 1623 → 1615 | 15 → 8 | 311 → 303 | 59.150 → 59.443 | 18/53 → 18/53 | 81/235 → 81/235.3 |

## Normalization / overlap

Full text duplicate extras L0/L1/L2 = {'L0': 257, 'L1': 274, 'L2': 338}. L2 là screening, không phải thêm review-copy đã xác nhận.
L2 empty-key rows: {'full': 31, 'train': 25, 'val': 3, 'test': 3}. Giữ collisions trong duplicate census; loại key rỗng khỏi overlap, gắn flag riêng.

| Level | Pair | Shared keys | Rows left | Rows right | Cross-pairs |
|---|---|---:|---:|---:|---:|
| L0 | train ↔ val | 18 | 148 | 38 | 1265 |
| L0 | train ↔ test | 18 | 148 | 30 | 795 |
| L0 | val ↔ test | 5 | 24 | 15 | 128 |
| L1 | train ↔ val | 19 | 156 | 39 | 1273 |
| L1 | train ↔ test | 19 | 155 | 31 | 802 |
| L1 | val ↔ test | 6 | 25 | 16 | 129 |
| L2 | train ↔ val | 22 | 179 | 48 | 1451 |
| L2 | train ↔ test | 21 | 173 | 37 | 910 |
| L2 | val ↔ test | 8 | 33 | 21 | 161 |

## Diagnostic test subsets

| Subset | Reviews | PRICE-Neutral support |
|---|---:|---:|
| original_test | 1623 | 1 |
| test_nonoverlap_l1 | 1592 | 0 |
| test_nonoverlap_l2 | 1585 | 0 |
| test_without_placeholder | 1615 | 1 |
| test_placeholder_only | 8 | 0 |

## Quyết định reporting

1. L1 là contamination chính; L0/L2 là sensitivity. L3 vẫn incomplete candidate screening, không được coi là exhaustive.
2. Cùng checkpoint: report original test, without-placeholder và nonoverlap diagnostics, kèm support/zero-support. Không retune từ các subset test.
3. Training-ablation bỏ placeholder chỉ trong train là run riêng, evaluation subsets cố định, không thay main benchmark.
4. Core joint metric không gồm OTHERS. Placeholder-only test có 0 core gold positives: recall N/A; FP vẫn phải báo nếu model hallucinate core aspects.
5. Chưa có predictions, nên chưa đo score inflation, hiệu quả loại NAN hoặc model superiority.

## Nguồn đã đối chiếu

- [Paper PACLIC 2022](https://aclanthology.org/2022.paclic-1.84.pdf)
- [VinAI PhoBERT: segmented input](https://github.com/VinAIResearch/PhoBERT)
- [scikit-learn F1: fixed labels / zero_division](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
