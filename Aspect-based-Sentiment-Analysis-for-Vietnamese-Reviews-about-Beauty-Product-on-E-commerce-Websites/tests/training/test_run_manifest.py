import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.training.run_manifest import (
    build_run_manifest,
    validate_run_manifest,
    verify_authorization,
)


ROOT = Path(__file__).resolve().parents[2]


def passing_observation(manifest):
    return {
        'git': {'commit': manifest['source']['git_commit'], 'dirty': False, 'dirty_sha256': None},
        'runtime': manifest['runtime']['expected'],
        'artifacts': manifest['integrity']['artifacts'],
        'integrity': manifest['integrity'],
        'training_source_sha256': manifest['source']['training_source_sha256'],
        'registration_contract': {
            'registration_id': manifest['registration']['registration_id'],
            'registration_sha256': manifest['registration']['registration_sha256'],
            'runtime_expected': manifest['runtime']['expected'],
            'registered_seeds_by_candidate': {
                manifest['candidate_id']: {'family': manifest['family'],
                                           'config': manifest['candidate_config'],
                                           'seeds': [42, 2026, 3407]}}},
        'assets': {'status': 'PASS', 'checked_files': ['synthetic-fixture'],
                   'phobert_revision': manifest['assets']['phobert_revision'],
                   'vncorenlp_revision': manifest['assets']['vncorenlp_revision'],
                   'java_release': manifest['assets']['java_release']},
        'hardware': {'cuda_available': True, 'gpu_name': 'Synthetic GPU', 'vram_gib': 16.0},
        'registered_run_inventory': {'status': 'PASS', 'count': 0, 'maximum': 18, 'conflicts': []},
        'checkpoint_path_exists': False,
    }


class RunManifestTests(unittest.TestCase):
    def test_manifest_binds_registered_run_and_never_pre_authorizes(self):
        manifest = build_run_manifest(ROOT, 'BIGRU-01', 42, 'synthetic-run-001')
        self.assertEqual(manifest['candidate_id'], 'BIGRU-01')
        self.assertEqual(manifest['seed'], 42)
        self.assertEqual(manifest['registration']['registration_id'], 'absa_model_comparison_registration_v2')
        self.assertEqual(manifest['data_policy']['fit_splits'], ['train'])
        self.assertEqual(manifest['data_policy']['selection_split'], 'val')
        self.assertFalse(manifest['data_policy']['test_used_for_selection'])
        self.assertIsNone(manifest['launch_authorization'])

    def test_manifest_rejects_unregistered_candidate_and_seed(self):
        with self.assertRaisesRegex(ValueError, 'registered candidate'):
            build_run_manifest(ROOT, 'PHOBERT-99', 42, 'bad-candidate')
        with self.assertRaisesRegex(ValueError, 'registered seed'):
            build_run_manifest(ROOT, 'PHOBERT-01', 123, 'bad-seed')
        with self.assertRaisesRegex(ValueError, 'run_id'):
            build_run_manifest(ROOT, 'TFIDF-01', 42, '..')

    def test_manifest_supports_configurable_durable_artifact_root(self):
        manifest = build_run_manifest(ROOT, 'TFIDF-01', 42, 'drive-run', '/content/drive/MyDrive/absa-runs')
        self.assertEqual(manifest['checkpoint_policy']['registry_directory'], '/content/drive/MyDrive/absa-runs')
        self.assertEqual(manifest['checkpoint_policy']['artifact_directory'],
                         '/content/drive/MyDrive/absa-runs/drive-run/checkpoints')

    def test_validation_blocks_runtime_and_dirty_checkout_mismatch(self):
        manifest = build_run_manifest(ROOT, 'TFIDF-01', 42, 'synthetic-run-002')
        observed = passing_observation(manifest)
        observed['git']['dirty'] = True
        observed['runtime'] = dict(observed['runtime'], python='3.13.9')
        report = validate_run_manifest(manifest, observed)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertTrue(any('dirty' in reason.lower() for reason in report['blockers']))
        self.assertTrue(any('python' in reason.lower() for reason in report['blockers']))

    def test_neural_validation_requires_pinned_assets_and_registered_gpu(self):
        manifest = build_run_manifest(ROOT, 'BIGRU-02', 2026, 'synthetic-run-003')
        observed = passing_observation(manifest)
        observed['assets'] = {'status': 'BLOCKED', 'checked_files': []}
        observed['hardware'] = {'cuda_available': False, 'gpu_name': None, 'vram_gib': 0}
        report = validate_run_manifest(manifest, observed)
        self.assertTrue(any('asset' in reason.lower() for reason in report['blockers']))
        self.assertTrue(any('gpu' in reason.lower() for reason in report['blockers']))

    def test_cpu_tfidf_validation_does_not_require_gpu_or_model_assets(self):
        manifest = build_run_manifest(ROOT, 'TFIDF-02', 3407, 'synthetic-run-004')
        observed = passing_observation(manifest)
        observed['assets'] = {'status': 'NOT_REQUIRED', 'checked_files': []}
        observed['hardware'] = {'cuda_available': False, 'gpu_name': None, 'vram_gib': 0}
        report = validate_run_manifest(manifest, observed)
        self.assertEqual(report['status'], 'PASS')

    def test_authorization_is_explicit_and_bound_to_manifest(self):
        manifest = build_run_manifest(ROOT, 'TFIDF-01', 42, 'synthetic-run-005')
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / 'run_manifest.json'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'authorization'):
                verify_authorization(manifest_path, 'synthetic-run-005')
            authorization = {
                'run_id': manifest['run_id'],
                'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
            (Path(directory) / 'run_authorization.json').write_text(json.dumps(authorization), encoding='utf-8')
            self.assertEqual(verify_authorization(manifest_path, 'synthetic-run-005')['run_id'], manifest['run_id'])
            manifest['seed'] = 2026
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_authorization(manifest_path, 'synthetic-run-005')


if __name__ == '__main__':
    unittest.main()
