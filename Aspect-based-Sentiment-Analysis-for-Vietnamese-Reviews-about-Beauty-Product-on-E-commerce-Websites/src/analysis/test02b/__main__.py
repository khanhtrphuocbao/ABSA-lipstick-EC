"""Recompute sealed prediction metrics and produce TEST-02B tables."""

import argparse

from src.analysis.common import ROOT, digest, write_once
from .core import CANDIDATES, SEEDS, SENTIMENTS
from .pipeline import close, run_analysis, save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    return run_analysis()


if __name__ == "__main__":
    main()
