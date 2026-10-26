"""Registered seed configuration; GPU bitwise identity is not asserted."""
import os
import random
import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
REGISTERED_SEEDS = tuple(json.loads((_ROOT / 'protocols/experiments/experiment_registration_v1.json').read_text())['seeds'])
STARTUP_HASH_SEED = os.environ.get('PYTHONHASHSEED')


def seed_everything(seed, deterministic_cuda=True, require_hash_seed=False):
    if seed not in REGISTERED_SEEDS:
        raise ValueError(f'Seed {seed} is not registered')
    effective = STARTUP_HASH_SEED == str(seed)
    if require_hash_seed and not effective:
        raise RuntimeError('Start a fresh process with PYTHONHASHSEED set to this registered seed')
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
    except ImportError as exc:
        raise ImportError('Registered training needs numpy and torch from modeling_requirements_v1.lock') from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    elif hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    cudnn = getattr(torch.backends, 'cudnn', None)
    if cudnn is not None and hasattr(cudnn, 'conv') and hasattr(cudnn.conv, 'fp32_precision'):
        cudnn.conv.fp32_precision = 'ieee'
        if hasattr(cudnn, 'rnn') and hasattr(cudnn.rnn, 'fp32_precision'):
            cudnn.rnn.fp32_precision = 'ieee'
    elif cudnn is not None:
        cudnn.allow_tf32 = False
    if deterministic_cuda and torch.cuda.is_available() and not os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be set before CUDA context creation')
    return {'seed': seed, 'python_hash_seed_effective': effective,
            'python_version': sys.version.split()[0], 'torch_deterministic_algorithms': True,
            'torch_threads': 1, 'cuda_bitwise_determinism_claimed': False}
