"""Registered fit primitives; deliberately no dataset discovery or auto-launch entrypoint."""
import math
from pathlib import Path

try:
    from ..absa_protocol import evaluate_rows
    from .checkpoint import (load_checkpoint, restore_torch_state,
                             save_checkpoint, serialize_torch_state)
    from .decoding import decode_probabilities, registered_configs, select_global_threshold
    from .losses import compute_multitask_loss
    from .models.tfidf_lr import TfidfLogisticABSA
except ImportError:
    from absa_protocol import evaluate_rows
    from training.checkpoint import (load_checkpoint, restore_torch_state,
                                     save_checkpoint, serialize_torch_state)
    from training.decoding import decode_probabilities, registered_configs, select_global_threshold
    from training.losses import compute_multitask_loss
    from training.models.tfidf_lr import TfidfLogisticABSA


def train_one_epoch(model, batches, optimizer, device='cpu', gradient_clip_norm=1.0,
                    gradient_accumulation=1, scheduler=None):
    import torch
    if gradient_accumulation < 1:
        raise ValueError('gradient_accumulation must be positive')
    try:
        number_of_batches = len(batches)
    except TypeError as exc:
        raise ValueError('Training batches must be a sized, repeatable loader') from exc
    if number_of_batches == 0:
        raise ValueError('Cannot train on an empty batch iterator')
    model.train()
    totals = {'loss': 0.0, 'aspect': 0.0, 'sentiment': 0.0, 'batches': 0}
    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(batches):
        inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
        targets = {key: value.to(device) for key, value in batch['targets'].items()}
        output = model(**inputs)
        loss, parts = compute_multitask_loss(output.aspect_logits, output.sentiment_logits,
                                            targets['aspect_targets'], targets['sentiment_targets'],
                                            targets['sentiment_mask'])
        group_size = min(gradient_accumulation,
                         number_of_batches - (index // gradient_accumulation) * gradient_accumulation)
        (loss / group_size).backward()
        if (index + 1) % gradient_accumulation == 0 or index + 1 == number_of_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        totals['loss'] += float(parts['total'].detach().cpu())
        totals['aspect'] += float(parts['aspect'].detach().cpu())
        totals['sentiment'] += float(parts['sentiment'].detach().cpu())
        totals['batches'] += 1
    return {key: value / totals['batches'] for key, value in totals.items() if key != 'batches'} | {'batches': totals['batches']}


class _RecurrentDataset:
    def __init__(self, rows, token_sequences, vocabulary):
        import torch
        from .targets import encode_batch
        if len(rows) != len(token_sequences) or not rows:
            raise ValueError('Rows and segmented token sequences must be nonempty and aligned')
        split = rows[0].get('source_split')
        if split not in ('train', 'val') or any(row.get('source_split') != split for row in rows):
            raise ValueError('Recurrent dataset must contain rows from one official train/val split')
        self.ids = [torch.tensor(vocabulary.encode(tokens, max_length=128), dtype=torch.long)
                    for tokens in token_sequences]
        targets = encode_batch(rows)
        self.targets = {key: torch.as_tensor(value) for key, value in targets.items()}
        self.split = split
        self.validation_rows = rows if split == 'val' else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item = {'input_ids': self.ids[index],
                'aspect_targets': self.targets['aspect_targets'][index],
                'sentiment_targets': self.targets['sentiment_targets'][index],
                'sentiment_mask': self.targets['sentiment_mask'][index]}
        if self.validation_rows is not None:
            item['gold_row'] = self.validation_rows[index]
        return item


def make_recurrent_loader(rows, token_sequences, vocabulary, batch_size, seed, shuffle):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    dataset = _RecurrentDataset(rows, token_sequences, vocabulary)
    from .reproducibility import REGISTERED_SEEDS
    if seed not in REGISTERED_SEEDS:
        raise ValueError('DataLoader seed is not registered')
    if batch_size != 32:
        raise ValueError('Registered recurrent batch size is 32')
    if shuffle != (dataset.split == 'train'):
        raise ValueError('Shuffle is enabled for train only and disabled for validation')
    generator = torch.Generator().manual_seed(seed)

    def collate(examples):
        ids = [item['input_ids'] for item in examples]
        batch = {'inputs': {'input_ids': pad_sequence(ids, batch_first=True, padding_value=0)},
                 'targets': {key: torch.stack([item[key] for item in examples])
                             for key in ('aspect_targets', 'sentiment_targets', 'sentiment_mask')}}
        if 'gold_row' in examples[0]:
            batch['gold_rows'] = [item['gold_row'] for item in examples]
        return batch

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       generator=generator if shuffle else None, num_workers=0,
                                       collate_fn=collate)


class _PhoBERTDataset:
    def __init__(self, rows, segmented_texts):
        import torch
        from .targets import encode_batch
        if not rows or len(rows) != len(segmented_texts):
            raise ValueError('PhoBERT rows and segmented texts must be nonempty and aligned')
        self.split = rows[0].get('source_split')
        if self.split not in ('train', 'val') or any(row.get('source_split') != self.split for row in rows):
            raise ValueError('PhoBERT dataset must contain official train or validation rows only')
        self.texts = list(segmented_texts)
        self.rows = rows if self.split == 'val' else None
        target_arrays = encode_batch(rows)
        self.targets = {key: torch.as_tensor(value) for key, value in target_arrays.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        item = {'segmented_text': self.texts[index],
                **{key: value[index] for key, value in self.targets.items()}}
        if self.rows is not None:
            item['gold_row'] = self.rows[index]
        return item


def make_phobert_loader(rows, segmented_texts, tokenizer, batch_size, seed, shuffle):
    import torch
    from .preprocessing import encode_phobert_batch
    from .reproducibility import REGISTERED_SEEDS
    dataset = _PhoBERTDataset(rows, segmented_texts)
    if seed not in REGISTERED_SEEDS:
        raise ValueError('DataLoader seed is not registered')
    if batch_size != 16:
        raise ValueError('Registered PhoBERT micro-batch size is 16')
    if shuffle != (dataset.split == 'train'):
        raise ValueError('Shuffle is enabled for train only and disabled for validation')
    generator = torch.Generator().manual_seed(seed)

    def collate(examples):
        encoded = encode_phobert_batch([item['segmented_text'] for item in examples],
                                       tokenizer, already_segmented=True)
        batch = {'inputs': encoded,
                 'targets': {key: torch.stack([item[key] for item in examples])
                             for key in ('aspect_targets', 'sentiment_targets', 'sentiment_mask')}}
        if 'gold_row' in examples[0]:
            batch['gold_rows'] = [item['gold_row'] for item in examples]
        return batch

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       generator=generator if shuffle else None, num_workers=0,
                                       collate_fn=collate)


def fit_tfidf_validation(train_rows, validation_rows, candidate_id, seed=None,
                         progress_callback=None):
    """Fit one registered lexical candidate on train and select its threshold on val."""
    from .decoding import decode_probabilities, select_global_threshold
    if not train_rows or any(row.get('source_split', 'train') != 'train' for row in train_rows):
        raise ValueError('TF-IDF fitting input must be official train rows only')
    if not validation_rows or any(row.get('source_split') != 'val' for row in validation_rows):
        raise ValueError('Threshold selection input must be official validation rows only')
    if seed is not None:
        from .reproducibility import REGISTERED_SEEDS
        if seed not in REGISTERED_SEEDS:
            raise ValueError('TF-IDF seed is not registered')
    model = TfidfLogisticABSA(candidate_id, random_state=seed).fit(train_rows,
                                                                  progress_callback=progress_callback)
    aspect, conditional = model.predict_probabilities([row['data'] for row in validation_rows])
    selected = select_global_threshold(validation_rows, aspect, conditional, 'val')
    predictions = [decode_probabilities(aspect[i], conditional[i], selected['threshold'], 'val',
                                        validation_rows[i]['source_row_index'])
                   for i in range(len(validation_rows))]
    metrics = evaluate_rows(validation_rows, predictions)
    return {'candidate_id': candidate_id, 'threshold': selected['threshold'],
            'validation_metrics': metrics, 'threshold_grid_metrics': selected, 'model': model}


def aggregate_candidate_results(seed_results, family):
    """Apply preregistered three-seed mean selection and tie-breaks."""
    configs = registered_configs()
    family_config = configs['families'][family]
    candidate_ids = [item['id'] for item in family_config['candidates']]
    expected_seeds = tuple(configs['seeds'])
    grouped = {candidate_id: {} for candidate_id in candidate_ids}
    for result in seed_results:
        candidate_id, seed = result.get('candidate_id'), result.get('seed')
        if candidate_id not in grouped or seed not in expected_seeds:
            raise ValueError('Unregistered candidate or seed in selection results')
        if seed in grouped[candidate_id]:
            raise ValueError('Duplicate candidate/seed validation result')
        grouped[candidate_id][seed] = result
    summary = {}
    for candidate_id, values in grouped.items():
        if tuple(sorted(values)) != tuple(sorted(expected_seeds)):
            raise ValueError(f'{candidate_id} must include all registered seeds {expected_seeds}')
        ordered = [values[seed] for seed in expected_seeds]
        metrics = {
            'seeds': list(expected_seeds),
            'mean_val_joint_macro_f1': sum(float(item['val_joint_macro_f1']) for item in ordered) / len(ordered),
            'mean_val_joint_micro_f1': sum(float(item['val_joint_micro_f1']) for item in ordered) / len(ordered)}
        if not all(math.isfinite(metrics[key]) for key in ('mean_val_joint_macro_f1', 'mean_val_joint_micro_f1')):
            raise ValueError('Candidate validation metrics must be finite')
        summary[candidate_id] = metrics
    selected = min(candidate_ids, key=lambda candidate_id: (
        -summary[candidate_id]['mean_val_joint_macro_f1'],
        -summary[candidate_id]['mean_val_joint_micro_f1'], candidate_id))
    return selected, summary


def build_neural_optimizer(model, candidate_id, total_optimizer_steps):
    import torch
    if type(total_optimizer_steps) is not int or total_optimizer_steps < 1:
        raise ValueError('total_optimizer_steps must be a positive integer')
    registration = registered_configs()
    if candidate_id.startswith('BIGRU-'):
        family = registration['families']['bigru_conv1d']
        if candidate_id not in {item['id'] for item in family['candidates']}:
            raise ValueError('Unknown registered BiGRU candidate')
        config = family['optimizer']
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        return optimizer, None, family
    if candidate_id.startswith('PHOBERT-'):
        family = registration['families']['phobert']
        candidate = next((item for item in family['candidates'] if item['id'] == candidate_id), None)
        if candidate is None:
            raise ValueError('Unknown registered PhoBERT candidate')
        config = family['optimizer']
        no_decay = tuple(config['no_decay'])
        decay_parameters, no_decay_parameters = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay_parameters if any(name.endswith(suffix) for suffix in no_decay)
             else decay_parameters).append(parameter)
        optimizer = torch.optim.AdamW([
            {'params': decay_parameters, 'weight_decay': config['weight_decay']},
            {'params': no_decay_parameters, 'weight_decay': 0.0}],
            lr=candidate['lr'], betas=tuple(config['betas']), eps=config['eps'])
        warmup_steps = max(1, math.ceil(total_optimizer_steps * .10))

        def schedule(step):
            if step < warmup_steps:
                return float(step + 1) / warmup_steps
            decay_steps = max(1, total_optimizer_steps - warmup_steps)
            return max(0.0, float(total_optimizer_steps - step) / decay_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return optimizer, scheduler, family
    raise ValueError('Candidate is not a registered neural model')


def fit_neural_candidate(model, train_batches, validation_batches, optimizer, scheduler,
                         candidate_id, seed, checkpoint_dir, metadata, device='cuda',
                         on_epoch_end=None):
    """Registered guarded fit loop; rejects CPU and non-validation checkpoint selection."""
    import numpy as np
    import torch
    try:
        from ..absa_protocol import CORE_ASPECTS, SENTIMENTS
        from .decoding import decode_probabilities, select_global_threshold
        from .reproducibility import REGISTERED_SEEDS
    except ImportError:
        from absa_protocol import CORE_ASPECTS, SENTIMENTS
        from training.decoding import decode_probabilities, select_global_threshold
        from training.reproducibility import REGISTERED_SEEDS
    if seed not in REGISTERED_SEEDS:
        raise ValueError('Seed is not registered')
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Registered neural fitting requires the authorized CUDA launch; CPU is diagnostics only')
    properties = torch.cuda.get_device_properties(torch.device(device))
    if properties.total_memory < 16 * 1024**3:
        raise RuntimeError('Registered neural fitting requires one CUDA GPU with at least 16 GiB VRAM')
    from .reproducibility import STARTUP_HASH_SEED
    if STARTUP_HASH_SEED != str(seed):
        raise RuntimeError('Launch a fresh process with PYTHONHASHSEED set before model initialization')
    if not torch.are_deterministic_algorithms_enabled() or torch.get_num_threads() != 1:
        raise RuntimeError('Call registered deterministic seed setup before constructing/fitting the model')
    if torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic:
        raise RuntimeError('cuDNN deterministic settings do not match the experiment registration')
    matmul_precision = getattr(torch.backends.cuda.matmul, 'fp32_precision', None)
    if matmul_precision is not None and matmul_precision != 'ieee':
        raise RuntimeError('TF32 must be disabled for the registered FP32 neural run')
    conv_precision = getattr(torch.backends.cudnn.conv, 'fp32_precision', None)
    if conv_precision is not None and conv_precision != 'ieee':
        raise RuntimeError('cuDNN TF32 must be disabled for the registered FP32 neural run')
    registration = registered_configs()
    if candidate_id.startswith('BIGRU-'):
        family = registration['families']['bigru_conv1d']
        accumulation = family['gradient_accumulation']
        maximum_epochs, patience = family['max_epochs'], family['patience']
    elif candidate_id.startswith('PHOBERT-'):
        family = registration['families']['phobert']
        accumulation = family['gradient_accumulation']
        maximum_epochs, patience = family['max_epochs'], family['patience']
    else:
        raise ValueError('Candidate is not a registered neural model')
    if candidate_id not in {item['id'] for item in family['candidates']}:
        raise ValueError('Candidate is not present in the experiment registration')
    model.to(device)
    history, best = [], None
    no_primary_improvement = 0
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, maximum_epochs + 1):
        train_metrics = train_one_epoch(model, train_batches, optimizer, device=device,
                                        gradient_clip_norm=family['gradient_clip_norm'],
                                        gradient_accumulation=accumulation, scheduler=scheduler)
        model.eval()
        gold_rows, aspect_probabilities, sentiment_probabilities = [], [], []
        with torch.no_grad():
            for batch in validation_batches:
                rows = batch.get('gold_rows')
                if not rows or any(row.get('source_split') != 'val' for row in rows):
                    raise ValueError('Checkpoint selection may use official validation rows only')
                inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
                output = model(**inputs)
                aspect_probabilities.append(torch.sigmoid(output.aspect_logits).cpu().numpy())
                sentiment_probabilities.append(torch.softmax(output.sentiment_logits, dim=-1).cpu().numpy())
                gold_rows.extend(rows)
        if not gold_rows:
            raise ValueError('Validation rows are empty')
        aspect_probabilities = np.concatenate(aspect_probabilities, axis=0)
        sentiment_probabilities = np.concatenate(sentiment_probabilities, axis=0)
        threshold_result = select_global_threshold(gold_rows, aspect_probabilities,
                                                   sentiment_probabilities, 'val')
        threshold = threshold_result['threshold']
        predictions = [decode_probabilities(aspect_probabilities[i], sentiment_probabilities[i],
                                            threshold, 'val', row['source_row_index'])
                       for i, row in enumerate(gold_rows)]
        validation_metrics = evaluate_rows(gold_rows, predictions)
        primary, micro = validation_metrics['joint']['macro_f1'], validation_metrics['joint']['micro']['f1']
        history.append({'epoch': epoch, 'train': train_metrics,
                        'val_joint_macro_f1': primary, 'val_joint_micro_f1': micro,
                        'threshold': threshold})
        if on_epoch_end is not None:
            on_epoch_end(dict(history[-1]))
        primary_improved = best is None or primary > best['macro']
        no_primary_improvement = 0 if primary_improved else no_primary_improvement + 1
        better_checkpoint = best is None or (primary, micro, -epoch) > (best['macro'], best['micro'], -best['epoch'])
        if better_checkpoint:
            epoch_metadata = {**metadata, 'candidate_id': candidate_id, 'seed': seed,
                              'epoch': epoch, 'threshold': threshold,
                              'validation_metrics': {'joint_macro_f1': primary, 'joint_micro_f1': micro}}
            path = checkpoint_dir / f'{candidate_id}-seed{seed}-epoch{epoch:02d}.json'
            checkpoint_hash = save_checkpoint(path, serialize_torch_state(model.state_dict()), epoch_metadata)
            best = {'epoch': epoch, 'macro': primary, 'micro': micro, 'threshold': threshold,
                    'checkpoint_path': str(path), 'checkpoint_sha256': checkpoint_hash,
                    'validation_metrics': validation_metrics}
        if no_primary_improvement >= patience:
            break
    if best is None:
        raise RuntimeError('No validation checkpoint was selected')
    best_bytes, _ = load_checkpoint(best['checkpoint_path'], expected_sha256=best['checkpoint_sha256'])
    model.load_state_dict(restore_torch_state(best_bytes, map_location=device))
    return {'candidate_id': candidate_id, 'seed': seed, 'best': best,
            'history': history, 'selection_split': 'val'}
