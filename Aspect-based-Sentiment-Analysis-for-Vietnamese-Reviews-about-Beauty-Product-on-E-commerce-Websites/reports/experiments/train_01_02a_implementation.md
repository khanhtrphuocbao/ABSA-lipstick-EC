# TRAIN-01 / TRAIN-02A Implementation Report

Date: 2026-10-25  
Scope: registered trainer infrastructure, synthetic CPU verification, and a pinned-commit Colab bootstrap. No real training was run.

## Status

| Work item | Status | Evidence / remaining boundary |
|---|---|---|
| TRAIN-01 trainer implementation | **Complete** | Three model adapters, train/validation data paths, target/loss/decoding, validation selection, checkpoint/export, and synthetic CPU tests are implemented. |
| TRAIN-02A Colab bootstrap | **Complete** | Nine ordered cells are valid JSON and their Python source parses. Portable hashes pass locally. The notebook was not executed on Colab. |
| TRAIN-02B GPU smoke | **Pending** | No CUDA runtime was used; target GPU, driver, CUDA packages, VRAM, and deterministic GPU behavior remain unverified. |
| TRAIN-03 run manifest / launch gate | **Pending** | Exact run manifest, resume policy, and explicit launch authorization remain to be finalized before registered fitting. |
| TRAIN-04 registered experiments | **Not started** | No registered candidate/seed fit or official test evaluation occurred. |

## Implementation

- `src/training/dataset.py`, `preprocessing.py`, `targets.py`, and `vocabulary.py` preserve split/row identity outside model inputs. Blank labels remain masked; `OTHERS` is an independent detection target and may coexist with core labels. Vocabulary fitting rejects non-train sources.
- `src/training/models/tfidf_lr.py` implements TFIDF-01 and TFIDF-02 from the registration, fits one shared train-only representation, fits eight binary heads and seven 3-class OVR heads, validates class support/order, and aligns conditional probabilities to `positive, neutral, negative`.
- `src/training/models/bigru.py` implements both registered recurrent variants. Packed sequences prevent right padding from changing backward-GRU context; the Conv1D branch masks padded positions before max pooling. `factory.py` binds shapes and defaults to the registration.
- `src/training/models/phobert.py` uses an injected `RobertaModel`, the registered BOS representation, dropout, and 8 + 7×3 heads. `preprocessing.py` provides the pinned segmentation/tokenizer path, dynamic right padding, a 128-token limit, and BOS/EOS checks. No pretrained weights are loaded implicitly.
- `losses.py`, `decoding.py`, and `trainer.py` implement the registered BCE + masked CE loss, global threshold grid/tie-break through the frozen evaluator, optimizer settings, accumulation, validation checkpoint selection, early stopping, and 3-seed candidate aggregation. Neural fitting rejects CPU and requires the registered CUDA/VRAM and deterministic launch conditions.
- `checkpoint.py` provides write-once atomic checkpoint envelopes, Torch and sklearn state serialization, required metadata, current trainer-code fingerprint, and mandatory SHA-256 verification on restore. `prediction_export.py` writes the existing row-keyed JSONL schema and validates it through the existing evaluator.
- `src/portable_verification.py` checks frozen file/CSV hashes, registration artifacts, PhoBERT/VnCoreNLP revisions and fingerprints, and the historical exact-runtime receipt. It explicitly reports that portable hash verification does **not** reproduce Python 3.15.1 / Unicode 17.0.0 normalization.
- `notebooks/colab_training.ipynb` has the nine requested cells: runtime, repo/commit configuration, exact checkout, dataset hashes, environment diagnostics/install instructions, portable verification, optional Drive, synthetic smoke, and pending launch gates. No cell invokes a registered training fit or downloads pretrained weights.

## Verification

- Frozen preflight: `python3 src/preflight_validation.py --check` — PASS (source hashes, split counts, manifest and sealed artifacts).
- Exact-runtime gate: `python3 src/reproducibility_gate.py --output /private/tmp/train01_02a_reproducibility_gate.json` — PASS on Python 3.15.1 / Unicode 17.0.0; original 16 tests passed. Output was kept in `/private/tmp`; the existing receipt was not overwritten.
- TRAIN-01 tests: 25/25 PASS on Python 3.13.9 / Unicode 15.1.0 with torch 2.9.0, scikit-learn 1.7.2, and transformers 4.57.1.
- CPU smoke: `python -m src.training.synthetic_smoke --cpu-only` — PASS. It fit TF-IDF on generated rows, ran one CPU optimizer step each for BiGRU and BiGRU+Conv1D, and checked a small random-config Roberta/PhoBERT adapter. The smoke reports `official_csvs_read=false`, `pretrained_weights_loaded=false`, and `actual_dataset_training=false`.
- Registration: `python3 src/register_experiments.py --verify` — PASS; registration remains 3 families × 2 candidates × 3 seeds (18 maximum planned fits).
- Portable integrity: PASS for all four official CSV hashes, frozen protocol/lock, registration artifacts, PhoBERT revision `01daacda68afe13d83023d16ec647239e344a1e6`, VnCoreNLP revision `62bbc58fe5d113c898eae112656be97dcf50b3a0`, and seven local tokenizer/segmenter/JDK asset files.
- Notebook: JSON parse and per-cell Python AST parse — PASS. No live Colab, Drive mount, or GPU execution was attempted.

Frozen checksums remained unchanged: freeze lock `16e9b76b6a71530d08704cadc2d1ddeddf54ca57615c35f8e650ccc57ac86ee8`; experiment registration `c49ea8ebf43928e67ef06e1f792fe792198328d70d4dd4b0bbfa11898f189a0d`. The original data protocol, evaluator, CSVs, registration, EDA notebooks, and prior evidence were not edited. No dependency lock or candidate configuration was changed.

## Independent verification commands

From repository root:

```bash
python3 src/preflight_validation.py --check
python3 -m unittest discover -s tests -v
/path/to/registered-python-3.13.9/bin/python -m unittest discover -s tests/training -v
/path/to/registered-python-3.13.9/bin/python -m src.training.synthetic_smoke --cpu-only
python3 src/register_experiments.py --verify
python3 src/portable_verification.py --root . --commit-sha <full-40-character-commit-sha> --assets-dir <pinned-assets-directory>
```

The original exact-runtime gate is separate from the modeling test command. Keep Python 3.15.1 / Unicode 17.0.0 for the frozen data gate and Python 3.13.9 / Unicode 15.1.0 for the registered modeling environment; the portable Colab check claims only immutable byte/hash verification.
