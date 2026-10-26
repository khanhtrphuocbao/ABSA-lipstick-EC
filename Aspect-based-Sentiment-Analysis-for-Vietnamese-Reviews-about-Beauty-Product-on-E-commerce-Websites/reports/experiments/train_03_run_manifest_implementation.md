# TRAIN-03 Run Manifest and Launch Gate

## Status

| Work item | Status | Evidence |
|---|---|---|
| Run manifest builder and CLI | Implemented | `src.training.run_manifest` |
| Candidate/seed and 18-run budget gates | Implemented | Reads registered v2 candidates/seeds and existing run manifests |
| Frozen artifact/source/runtime checks | Implemented | Hash-only checks for official CSVs; protocol, freeze locks, both registrations, historical Python 3.15.1 / Unicode 17.0.0 receipt, and training/evaluator source code |
| Neural asset/GPU gate | Implemented, currently unverified | Requires pinned asset fingerprints and one CUDA GPU with at least 16 GiB VRAM |
| Explicit authorization | Implemented as a separate write-once record | Bound to the exact run-manifest SHA and run ID |
| Actual trainer launch/resume | Pending TRAIN-04 | `launch-check` validates authorization and exits without starting a fit; optimizer/scheduler/RNG resume is not implemented |

The manifest records candidate config, seed, v1/v2 registration references, data byte hashes, train-only fit and validation-only selection, evaluation/threshold policies, code SHA-256, runtime pins and observed runtime, asset revisions, artifact path, and checkpoint/resume policy. It never converts a blank label into Neutral, reads CSV rows for a model, loads PhoBERT weights, or starts training.

Validation currently cannot authorize a real run until the pinned neural assets have been fingerprint-checked and target GPU details are available. The latest portable Colab result reported `asset_files_checked: []`; that is not an asset pass. The v2 package/runtime result is useful evidence for the Colab environment but does not replace the v1 historical exact data-gate receipt.

## Commands

Run from the project root. Choose a new unique run ID for each registered candidate/seed:

```bash
python -m src.training.run_manifest prepare \
  --candidate BIGRU-01 --seed 42 --run-id BIGRU-01-seed-42

python -m src.training.run_manifest validate \
  --manifest artifacts/runs/BIGRU-01-seed-42/run_manifest.json \
  --assets-dir /path/to/pinned/model-assets
```

For a TF-IDF candidate, omit `--assets-dir`; GPU is not required. A neural manifest stays BLOCKED until all asset fingerprints and the registered GPU requirement pass. Validation writes an immutable timestamped JSON observation beside the manifest.

The following commands are separate and do not train. `authorize` is only for a future, explicitly approved launch after a clean validation PASS. `launch-check` deliberately reports that the TRAIN-04 fitter is not implemented:

```bash
python -m src.training.run_manifest authorize \
  --manifest artifacts/runs/BIGRU-01-seed-42/run_manifest.json \
  --confirm-run-id BIGRU-01-seed-42 --assets-dir /path/to/pinned/model-assets

python -m src.training.run_manifest launch-check \
  --manifest artifacts/runs/BIGRU-01-seed-42/run_manifest.json \
  --run-id BIGRU-01-seed-42 --assets-dir /path/to/pinned/model-assets
```

`prepare`, `validate`, `authorize`, and `launch-check` contain no call to a trainer. Normal notebook Run All remains non-training.

## Verification boundary

The manifest tests exercise generated observation objects and a temporary authorization file. Existing preflight tests and the existing synthetic-model suite remain separate. No official CSV was fitted, no GPU was used, no pretrained weights were downloaded, and no experiment score was generated. A clean, pinned Python 3.13.16 / Unicode 15.1.0 checkout and TRAIN-02B GPU smoke are still required before TRAIN-04 can define and enable a fit launcher.
