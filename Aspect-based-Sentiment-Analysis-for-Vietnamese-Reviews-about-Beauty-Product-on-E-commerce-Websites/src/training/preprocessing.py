"""Model preprocessing adapters, separate from L0/L1/L2 audit normalization."""
import base64
import subprocess
import tempfile
from pathlib import Path

try:
    from ..modeling_preflight import prepare_text
except ImportError:
    from modeling_preflight import prepare_text


def prepare_texts(texts):
    if any(not isinstance(text, str) for text in texts):
        raise ValueError('Model inputs must be strings')
    return [prepare_text(text) for text in texts]


def segment_vietnamese(texts, assets_dir, repo_root=None):
    """Run the pinned VnCoreNLP bridge, framing each review independently."""
    from . import assets
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    assets_dir = Path(assets_dir)
    asset_report = assets.verify_assets(assets_dir, root, require_java_archive=False)
    java = Path(asset_report['java_executable'])
    jar = assets_dir / 'vncorenlp/VnCoreNLP-1.2.jar'
    bridge = root / 'src/VnCoreNLPBatch.java'
    if not java.is_file() or not jar.is_file() or not bridge.is_file():
        raise FileNotFoundError('Pinned JDK, VnCoreNLP jar, or segmentation bridge is missing')
    prepared = prepare_texts(texts)
    with tempfile.TemporaryDirectory(prefix='absa-segment-') as tmp:
        input_path, output_path = Path(tmp) / 'in.tsv', Path(tmp) / 'out.tsv'
        input_path.write_text(''.join(f'{i}\t{base64.b64encode(text.encode()).decode()}\n'
                                      for i, text in enumerate(prepared)), encoding='utf-8')
        subprocess.run([str(java), '-Xmx1g', '--class-path', str(jar), str(bridge),
                        str(input_path), str(output_path)], cwd=jar.parent, check=True, timeout=600,
                       capture_output=True, text=True)
        result = {}
        for line in output_path.read_text(encoding='utf-8').splitlines():
            index, value = line.split('\t', 1)
            index = int(index)
            if index in result:
                raise ValueError('Segmenter returned a duplicate row')
            result[index] = base64.b64decode(value).decode('utf-8')
    if set(result) != set(range(len(texts))):
        raise ValueError('Segmenter output does not cover every input row')
    return [result[i] for i in range(len(texts))]


def recurrent_tokens(segmented_text):
    return segmented_text.split()


def load_phobert_tokenizer(assets_dir):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError('PhoBERT tokenization requires transformers==4.57.1 from the registered lock') from exc
    from .assets import verify_assets
    verify_assets(assets_dir, require_java_archive=False)
    return AutoTokenizer.from_pretrained(Path(assets_dir) / 'phobert', use_fast=False,
                                         local_files_only=True)


def encode_phobert_batch(texts, tokenizer, *, already_segmented=False, assets_dir=None):
    import torch
    if not texts:
        raise ValueError('PhoBERT tokenization needs at least one review')
    if tokenizer.truncation_side != 'right':
        raise ValueError('Registered PhoBERT truncation side must be right')
    if getattr(tokenizer, 'bos_token_id', None) is None or getattr(tokenizer, 'eos_token_id', None) is None:
        raise ValueError('PhoBERT tokenizer must expose BOS and EOS token IDs')
    if already_segmented:
        prepared = list(texts)
    else:
        if assets_dir is None:
            raise ValueError('Pinned VnCoreNLP assets are required before tokenization')
        prepared = segment_vietnamese(texts, assets_dir)
    encoded = tokenizer(prepared, add_special_tokens=True, truncation=True, max_length=128,
                        padding=True, return_attention_mask=True, return_tensors='pt')
    input_ids, attention_mask = encoded['input_ids'], encoded['attention_mask']
    if (input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or
            input_ids.shape[0] != len(texts) or input_ids.shape[1] > 128):
        raise ValueError('PhoBERT tokenizer returned incompatible padded batch dimensions')
    if len(texts) and torch.any(input_ids[:, 0] != tokenizer.bos_token_id):
        raise ValueError('PhoBERT BOS token was not preserved')
    for i in range(input_ids.shape[0]):
        valid = int(attention_mask[i].sum().item())
        if valid < 2 or input_ids[i, valid - 1].item() != tokenizer.eos_token_id:
            raise ValueError(f'PhoBERT EOS token was not preserved after truncation at row {i}')
    return {'input_ids': input_ids, 'attention_mask': attention_mask}
