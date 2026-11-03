# ABSA Code Simplification Implementation Plan

> Use superpowers:executing-plans inline, with focused regression tests.

**Goal:** Simplify existing analysis without changing experiment behavior or commands.
**Architecture:** Shared byte-level IO in `src/analysis/common.py`; B/C processing in their own `pipeline.py`; thin compatible `__main__.py` entry points. Keep historical figure-generator sources byte-identical.
**Tech Stack:** Python standard library, unittest; Matplotlib 3.10.6 in an isolated reporting environment.
**Spec:** `docs/superpowers/specs/2026-11-03-code-simplification-design.md`

## Constraints and decisions

- No model fitting, inference, commits, pushes or upstream result writes.
- Preserve training, evaluator, protocol, data, all existing dirty changes.
- Work in the approved checkout: current analysis is untracked and missing from HEAD. Keep an independent before-copy in /private/tmp rather than checking out an older tree.
- Do not edit `src/analysis/test02d/{__main__,sources,figures}.py`: existing figure receipts bind their bytes. Its existing compatibility import remains valid.
- Preserve distinct null aggregation and bytes/JSON writing conventions; similarly named functions are not automatically equivalent.

## Review focus

Missing files; differing existing outputs; null CSV fields; CLI help triggering processing; legacy imports and historical figure freshness.

## Task 1 — Shared IO

- [x] Add tests for CSV byte conventions, null values, write-once collisions/idempotency and content digests. Watch missing-module failure.
- [x] Add `src/analysis/common.py`: ROOT, digest(path), csv_bytes(rows, fields=None, extrasaction='raise'), write_once(path,payload).
- [x] Verify new tests and existing quantitative tests.

## Task 2 — Processing and CLI boundaries

- [x] Add tests for safe `--help`, legacy helpers and behavior against sealed predictions.
- [x] Move B/C processing into `test02b/pipeline.py` and `test02c/pipeline.py`, preserving algorithms. Keep explicit compatibility exports in entry points.
- [x] Update supplement, closure, reporting imports to ordinary modules; expand compressed formatting. Keep serialized report text unchanged.
- [x] Verify protocol and all B/C/D tests; verify figure freshness and unchanged original inputs.

## Task 3 — Documentation and final checks

- [x] Rewrite root README around the current pipeline, preserving original paper attribution and notebook context.
- [x] Run CLI-help checks, all supported tests and byte comparisons for experimental inputs and generated document content.
- [x] Review scoped diff, preserve old results, and report any environment-related failures.

## Verification results

- Shared IO, safe help, compatibility and freshness: 9 tests PASS (including RED/GREEN for restored compatibility exports).
- Numeric regressions: all 9 B/C CSV tables match sealed bytes; processing is exercised in memory with only output IO replaced.
- Default discovery on Python 3.15: 55 tests PASS, 1 module import error (`test_test02d_figures`: Matplotlib is unavailable in this interpreter).
- Separate Python 3.11 reporting environment, Matplotlib 3.10.6: 17 tests PASS; no Unicode-runtime gate is bypassed.
- Figure freshness: CURRENT. All 240 protected files unchanged, including training, evaluator, datasets, protocols and experimental outputs.
- Black check: all 12 scoped Python files clean.
- Independent review: no blocking findings. Restored C entry-point digest/write_once exports after a failing compatibility test.
- No fitting, model inference, artifact regeneration, commit or push.
