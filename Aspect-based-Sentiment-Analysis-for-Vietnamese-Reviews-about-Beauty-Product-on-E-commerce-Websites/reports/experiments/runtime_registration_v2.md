# Registration v2 runtime amendment

## Decision

Registration v2 pins the modeling interpreter to CPython 3.13.16 / Unicode 15.1.0 to match the Colab runtime observed by the project owner. Registration v1, the Data & Evaluation Protocol v1, freeze lock, evaluator, and all v1 checksums remain unchanged.

## Scope

This is a runtime-only amendment. It inherits the six v1 candidates, seeds 42/2026/3407, threshold grid and tie-breaks, losses, checkpoint selection, metrics, data split rules, and 18-run maximum without changes. Package versions remain torch 2.9.0, scikit-learn 1.7.2, and transformers 4.57.1. The existing hash-locked package graph is reused; its v1 files are not rewritten.

## Verification status

- V1 integrity: verified by the existing v1 registration lock; v1 files were not edited.
- Colab CPU synthetic smoke: PASS on generated data only; no official CSVs, pretrained weights, or GPU used.
- V2 runtime/package gate: **PENDING**. Re-run Cell 1 and Cell 5 after installing the locked packages. The earlier captured Colab package versions (torch 2.11.0+cu130, scikit-learn 1.6.1, transformers 5.20.2) do not satisfy v2.
- TRAIN-02B GPU smoke: PENDING.
- TRAIN-03 run manifest, checkpoint/resume policy, code hash and explicit launch gate: PENDING.
- TRAIN-04 registered model fits: NOT STARTED.

No training was started as part of this amendment. The Colab notebook must pass the v2 environment check and the remaining TRAIN-03 gates before any registered fit.
