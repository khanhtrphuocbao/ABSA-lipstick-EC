# TEST-02D reporting environment

Create a separate Python 3.11 virtual environment and install `matplotlib==3.10.6` from `requirements-reporting.txt`. The reporting runner uses the Agg backend. Example from the repository root:

```sh
python3.11 -m venv /private/tmp/absa-test02d-reporting
/private/tmp/absa-test02d-reporting/bin/python -m pip install -r src/analysis/test02d/requirements-reporting.txt
env MPLCONFIGDIR=/private/tmp/absa-test02d-mpl /private/tmp/absa-test02d-reporting/bin/python -B -m src.analysis.test02d --validation-aggregate ../absa-run/validation_aggregate_assumed_equivalence.json
/private/tmp/absa-test02d-reporting/bin/python -B -m src.analysis.test02d --check-stale
```

The manifest records the Python, Matplotlib, backend and platform actually used. Output collisions are rejected unless bytes are identical. Figure 02 is blocked when the validation aggregate is missing or its SHA-256 differs from the selection freeze. The reporting environment is independent of the frozen modeling environment.
