"""Refactored processing must reproduce sealed numeric tables in memory."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch


class SealedAnalysisRegressionTests(unittest.TestCase):
    def check_tables(self, pipeline, filenames):
        directory = pipeline.ROOT / "reports/experiments/test_02"
        before = {name: (directory / name).read_bytes() for name in filenames}
        produced = {}

        def collect(path, payload):
            produced[Path(path).name] = payload

        # Replace only disk output: reading, alignment, scoring and aggregation
        # all use the real existing artifacts and evaluator, without model loads.
        with patch.object(pipeline, "write_once", side_effect=collect):
            with redirect_stdout(io.StringIO()):
                pipeline.run_analysis()

        for name, expected in before.items():
            with self.subTest(name=name):
                self.assertEqual(produced[name], expected)
                self.assertEqual((directory / name).read_bytes(), expected)

    def test_quantitative_tables_match_existing_results_byte_for_byte(self):
        from src.analysis.test02b import pipeline

        self.check_tables(
            pipeline,
            (
                "per_pair_metrics.csv",
                "per_pair_deltas.csv",
                "macro_f1_decomposition.csv",
            ),
        )

    def test_error_analysis_tables_match_existing_results_byte_for_byte(self):
        from src.analysis.test02c import pipeline

        self.check_tables(
            pipeline,
            (
                "error_taxonomy.csv",
                "confusion_summary.csv",
                "cross_model_error_comparison.csv",
                "linguistic_error_summary.csv",
                "representative_errors.csv",
                "diagnostic_subset_comparison.csv",
            ),
        )


if __name__ == "__main__":
    unittest.main()
