# TRAIN-04 Registered Runner

## Status

TRAIN-04 code is implemented for one explicitly authorized candidate/seed at a time. No registered fit was run while implementing it.

The runner:

- Revalidates the TRAIN-03 authorization, clean Git commit, v1/v2 registration hashes, v2 Python/package pins, frozen CSV hashes, model assets, and neural GPU requirement immediately before reading split rows.
- Loads only `data_train.csv` and `data_val.csv` as model inputs. It does not parse the full or test CSVs, and never evaluates on test.
- Uses the registered TF-IDF, BiGRU/Conv1D, or PhoBERT family settings and seed. TF-IDF is assigned the run seed for its LogisticRegression heads. PhoBERT weights are resolved from the registered repository revision only inside the explicit run command.
- Saves a write-once checkpoint, validation JSONL, run result, and append-only run events under the manifest's artifact directory. Existing checkpoint/resume state is rejected; optimizer/scheduler/RNG resume is not supported.
- Emits progress records during segmentation, TF-IDF head fitting, and neural epochs.
- Aggregates family selection only after both candidates have completed all three registered seeds, using the registered validation macro-F1, micro-F1, and candidate-ID tie-break. The summary command does not access test.

The Java adapter accepts the registered `jdk-17.0.21+9` runtime from `JAVA_HOME` or `PATH`, which supports Linux Colab. The PhoBERT tokenizer and VnCoreNLP files still require fingerprint verification. A future PhoBERT run downloads model weights only after the explicit authorized `run` command starts.

## Colab workflow

First pin Cell 2 to the commit containing this implementation. Run the existing integrity and runtime cells, then stage the registered tokenizer/segmenter assets and all four frozen CSVs. The Java 17.0.21+9 executable must be installed in the runtime. Choose an artifact root on Drive if durable storage is needed.

For neural candidates, restore only the registered tokenizer/segmenter files with `python -m src.training.restore_runtime_assets --assets /content/absa-assets`. The command checks each pinned SHA-256, refuses to overwrite mismatches, and does not fetch PhoBERT weights or a JDK. Install/select the exact registered Java runtime separately. The `--assets-dir` argument is required on **each** `validate`, `authorize`, and `run` command for BiGRU/PhoBERT; TF-IDF does not require it.

Prepare and validate one run. Repeat with a unique run ID for every candidate/seed pair:

```bash
python -m src.training.run_manifest prepare \
  --candidate TFIDF-01 --seed 42 --run-id TFIDF-01-seed-42 \
  --artifact-root /content/drive/MyDrive/absa-runs \
  --data-dir /content/absa-data

python -m src.training.run_manifest validate \
  --manifest /content/drive/MyDrive/absa-runs/TFIDF-01-seed-42/run_manifest.json \
  --data-dir /content/absa-data
```

After a validation PASS, write the separate authorization. This command still does not fit:

```bash
python -m src.training.run_manifest authorize \
  --manifest /content/drive/MyDrive/absa-runs/TFIDF-01-seed-42/run_manifest.json \
  --confirm-run-id TFIDF-01-seed-42 --data-dir /content/absa-data
```

The actual fit occurs only when the distinct TRAIN-04 command below is deliberately invoked in a fresh process. The environment variables must be present before Python starts:

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m src.training.registered_run run \
  --manifest /content/drive/MyDrive/absa-runs/TFIDF-01-seed-42/run_manifest.json \
  --run-id TFIDF-01-seed-42 --data-dir /content/absa-data
```

For BiGRU and PhoBERT, add `--assets-dir /content/absa-assets` to both validation and authorization, then pass it to the run command; the neural gate also requires one CUDA GPU with at least 16 GiB VRAM. PhoBERT's exact pretrained revision is loaded only after all gates pass.

After all six candidate/seed runs in a family are complete, summarize selection on validation:

```bash
python -m src.training.registered_run aggregate \
  --family tfidf_logistic \
  --artifact-root /content/drive/MyDrive/absa-runs
```

There is no Run All hook, no command that automatically launches all 18 runs, and no test-evaluation command in TRAIN-04. Each of the maximum 18 registered candidate/seed runs needs its own manifest, validation, authorization, and explicit `run` invocation.

## Verification boundary

Tests cover run authorization binding, train/validation-only split paths, startup seed/CUDA environment gates, Java release verification, TF-IDF seed wiring, and complete three-seed family aggregation. Existing synthetic model tests remain the only model fits run during verification. The official data has not been fitted, no pretrained PhoBERT weights were downloaded, and no GPU training was performed.

Implementation checks on the local development environment:

- Preflight suite: 16/16 PASS.
- Training suite: 41/41 PASS, all synthetic/unit checks.
- Python syntax compilation and all three CLI help commands: PASS.
- Colab notebook JSON parse: PASS (9 cells).
- All 19 artifacts referenced by registration v1/v2 locks: SHA-256 match.
- Local Python 3.12 is not the registered Python 3.13.16 / Unicode 15.1.0 modeling runtime, and this checkout is dirty; therefore the TRAIN-03 launch gate cannot pass here.

In Colab, first verify the pinned commit and registered environment; neural runs additionally require staged fingerprinted assets and the registered Java runtime. TRAIN-02B hardware smoke remains pending.
