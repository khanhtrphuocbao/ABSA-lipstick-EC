"""TEST-02C: read sealed predictions and produce descriptive error analysis."""

import argparse

from src.analysis.common import ROOT, digest, write_once
from .pipeline import (
    ASPECTS,
    CORE_ASPECTS,
    SENTIMENTS,
    CANDIDATES,
    SEEDS,
    SUBSETS,
    EXPECTED,
    csv_bytes,
    events_for,
    feature_flags,
    run_analysis,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    return run_analysis()


if __name__ == "__main__":
    main()
