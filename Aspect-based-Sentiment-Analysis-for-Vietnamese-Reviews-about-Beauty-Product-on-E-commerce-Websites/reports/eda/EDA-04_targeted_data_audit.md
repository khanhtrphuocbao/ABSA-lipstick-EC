# EDA-04 — Targeted Data Audit

## Scope and provenance

Read-only analysis of four source CSV files; no label corrections, row removal, new split or trained model.
Rare audit threshold: core aspect–sentiment train count ≤25 (set before selecting samples).
Source rows are zero-based parsed CSV record indices, not physical line numbers.
Sources retain the SHA-256 fingerprints printed in the notebook setup.

## 1. OTHERS versus core sentiments

Train all-8: Positive 71.27%, Neutral 14.57%, Negative 14.16%; IR 5.03.
Train core-7: Positive 78.09%, Neutral 6.40%, Negative 15.52%; IR 12.21.
Decision: retain 8 aspect labels, report sentiment on 7 core aspects and OTHERS/spam separately. Excluding OTHERS means excluding its label column from sentiment counts, not discarding whole reviews.

## 2. Extreme rare combinations

| Pair | Train | Validation | Test | Test non-overlap support |
|---|---:|---:|---:|---:|
| PRICE × Neutral | 22 | 4 | 1 | 0 |
| PRICE × Negative | 15 | 0 | 6 | 6 |
| PACKING × Neutral | 15 | 1 | 2 | 2 |


Census support is complete; the notebook records 9 deliberately chosen train observations plus one contaminated test sample. These are qualitative candidates, not an estimated annotation error rate.
PRICE-negative spans explicit expense, poor value and regret after a price drop. PRICE-neutral includes descriptive/uncertain prices, resale content and ambiguous evaluation. PACKING-neutral includes neutral mentions and mixed complaints; train row 11287 has an explicit packing praise cue despite its neutral label.
Action: adjudicate boundary/scope/aggregation rules before weighting rare labels. Validation has no PRICE-negative support; use train-only grouped sensitivity analysis if needed, and never select the loss from test.

## 3. OTHERS together with core aspects

Four mixed reviews occur in train (rows 1096, 1377, 3745, 10993), none in validation/test. All four retain SHIPPING-positive; row 1377 also has PACKING-positive. All four were inspected.
Action: do not implement a mutually exclusive OTHERS/core target or a hard OTHERS discard gate. Preserve core annotations and audit the definition of spam components versus whole-review spam.

## 4. Duplication and reporting

Full has 251 duplicate rows beyond the first occurrence. Train has 6 conflicting normalized text keys involving 48 rows.

Test has 31/1623 (1.91%) review rows overlapping train or validation under lowercase/trim/whitespace normalization; the non-overlap subset has 1592 rows.

The only test PRICE-neutral review (row 720) is an exact text-and-label-vector duplicate of train row 12120. Excluding overlap reduces this class's test support from 1 to 0. Report original-test results and clearly named overlap/non-overlap diagnostics, with per-class support. Keep the fixed 21 core pair labels for primary Macro-F1; recall at zero gold support is N/A. An observed-support macro, if included, is a separate secondary metric.

Contamination is observed; its effect on model scores is unmeasured because no predictions exist. A non-overlap subset is not a newly validated clean benchmark and does not replace the original test silently.

## 5. Updated methodology and limitations

1. Freeze benchmark hashes, label space, metric units, zero-support handling and reporting protocol.
2. Audit rare/conflict/blank annotations and mixed OTHERS/core rules. Corrections require a separate version and adjudication protocol.
3. Compare lexical baselines with contextual models; unweighted loss first, then validated weighting/focal ablations.
4. Report joint Macro/Micro/Weighted-F1, per-aspect/per-sentiment metrics and Negative precision/recall with support; OTHERS detection separately.
5. Preserve raw nonstandard text in the baseline; tokenizer/max_length decisions still require train token-length analysis.

Limitations: sparse minority support and zero-support validation; heterogeneous Neutral semantics; annotation inconsistency candidates; observed exact/normalized contamination; no near-duplicate, product/user/time leakage audit; selective qualitative samples and no adjudicated error rate. Test was already inspected descriptively. No model superiority, causal shortcut or score inflation has been measured. The paper's corpus contains 9 lipstick products; broader business generalization needs an external or product/time-held-out benchmark and corresponding metadata.

Paper reference: [Tran et al., PACLIC 2022, §3 and Table 1](https://aclanthology.org/2022.paclic-1.84.pdf).

## Input fingerprints

- `full_data.csv`: `3bda8deecc84d94fff526ad092cadc1dc4a4487e6d1a8c54871e53da95ec780e`
- `data_train.csv`: `19449cceea317269304464ef7c6c72d5dc70548c6006686d1506e1880bfb9976`
- `data_val.csv`: `7bbc93fad8db9c78a83f75800119cad65bf65b8f0241c36228d7cbe13d1535b3`
- `data_test.csv`: `90d1bf0b64105c89346f60d974c5c72c0208c70458c40d5b089ac48486d43d08`
