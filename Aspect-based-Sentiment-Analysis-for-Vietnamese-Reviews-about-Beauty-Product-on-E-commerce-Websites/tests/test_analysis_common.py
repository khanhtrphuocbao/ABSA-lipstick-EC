"""Contracts that must survive moving shared analysis utilities."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest


class AnalysisIOTests(unittest.TestCase):
    def test_csv_preserves_nulls_unicode_column_order_and_line_endings(self):
        from src.analysis.common import csv_bytes

        rows = [{"aspect": "mùi", "f1": None, "support": 0}]
        self.assertEqual(csv_bytes(rows), "aspect,f1,support\r\nmùi,,0\r\n".encode("utf-8"))

    def test_csv_ignores_extra_fields_only_when_requested(self):
        from src.analysis.common import csv_bytes

        rows = [{"value": 2, "private": "not exported"}]
        with self.assertRaises(ValueError):
            csv_bytes(rows, ["value"])
        self.assertEqual(csv_bytes(rows, ["value"], extrasaction="ignore"), b"value\r\n2\r\n")
        self.assertEqual(csv_bytes([], ["value"]), b"value\r\n")

    def test_digest_reads_bytes_without_mutation_and_rejects_missing_file(self):
        from src.analysis.common import digest

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            path.write_bytes(b"abc")
            self.assertEqual(
                digest(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            self.assertEqual(path.read_bytes(), b"abc")
            with self.assertRaises(FileNotFoundError):
                digest(Path(tmp) / "missing")

    def test_write_once_preserves_identical_file_and_rejects_overwrite(self):
        from src.analysis.common import write_once

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp) / "nested" / "result.json"
            write_once(path, b'{"value":null}\n')
            timestamp = path.stat().st_mtime_ns
            write_once(path, b'{"value":null}\n')
            self.assertEqual(path.stat().st_mtime_ns, timestamp)
            with self.assertRaises(FileExistsError):
                write_once(path, b'{"value":0}\n')
            self.assertEqual(path.read_bytes(), b'{"value":null}\n')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_competing_writes_cannot_overwrite_the_winning_complete_payload(self):
        from src.analysis.common import write_once

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result"
            payloads = [b"a" * 1000, b"b" * 1000]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(write_once, path, data) for data in payloads]
            errors = [f.exception() for f in futures if f.exception()]
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], FileExistsError)
            self.assertIn(path.read_bytes(), payloads)
            self.assertEqual(list(Path(tmp).iterdir()), [path])


class AnalysisEntryPointTests(unittest.TestCase):
    def test_help_exits_without_running_analysis(self):
        import importlib
        from unittest.mock import patch

        for name in ("test02b", "test02c"):
            module = importlib.import_module(f"src.analysis.{name}.__main__")
            with self.subTest(name=name), redirect_stdout(io.StringIO()) as output:
                # Only the expensive analysis boundary is replaced; argparse is real.
                with patch.object(
                    module,
                    "run_analysis",
                    create=True,
                    side_effect=AssertionError("analysis ran on --help"),
                ):
                    with self.assertRaises(SystemExit) as exit_status:
                        module.main(["--help"])
                self.assertEqual(exit_status.exception.code, 0)
                self.assertIn("usage:", output.getvalue())

    def test_legacy_entry_point_exports_preserve_consumers(self):
        from src.analysis.common import digest, write_once
        from src.analysis.test02b.__main__ import digest as old_digest
        from src.analysis.test02b.__main__ import write_once as old_write
        from src.analysis.test02c.__main__ import csv_bytes, events_for
        from src.analysis.test02c.__main__ import digest as error_digest
        from src.analysis.test02c.__main__ import write_once as error_write

        self.assertIs(old_digest, digest)
        self.assertIs(old_write, write_once)
        self.assertIs(error_digest, digest)
        self.assertIs(error_write, write_once)
        self.assertEqual(csv_bytes([{"x": None, "extra": 1}], ["x"]), b'x\r\n""\r\n')
        self.assertTrue(callable(events_for))


class FigureStatusTests(unittest.TestCase):
    def test_historical_figures_keep_their_real_status(self):
        import json
        from src.analysis.common import ROOT
        from src.analysis.figure_status import stale

        manifest = json.loads(
            (ROOT / "reports/experiments/test_02/figure_manifest.json").read_text()
        )
        self.assertEqual(stale(manifest), [])

    def test_changes_to_generator_source_or_output_are_detected(self):
        from src.analysis.common import digest
        from src.analysis.figure_status import generator_sha, stale

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            generator = directory / "generator.py"
            source = directory / "values.csv"
            output = directory / "figure.png"
            generator.write_text("version = 1\n")
            source.write_bytes(b"value\n1\n")
            output.write_bytes(b"sealed plot")
            manifest = {
                "figures": [
                    {
                        "figure_id": "FIG-01",
                        "generator_sha256": generator_sha(directory),
                        "sources": [{"path": str(source), "sha256": digest(source)}],
                        "outputs": [{"path": str(output), "sha256": digest(output)}],
                    }
                ]
            }
            self.assertEqual(stale(manifest, generator_directory=directory), [])
            source.write_bytes(b"value\n2\n")
            self.assertTrue(stale(manifest, generator_directory=directory))
            source.write_bytes(b"value\n1\n")
            output.unlink()
            self.assertTrue(stale(manifest, generator_directory=directory))
            output.write_bytes(b"sealed plot")
            generator.write_text("version = 2\n")
            self.assertTrue(stale(manifest, generator_directory=directory))


if __name__ == "__main__":
    unittest.main()
