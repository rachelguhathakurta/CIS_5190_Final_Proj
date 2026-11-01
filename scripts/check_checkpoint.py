"""Audit saved weights and compare training/submission inference locally."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import my_model as model_module
from my_model import Model, NormalizedRegressor
from preprocess import compute_norm_stats, prepare_data
from eval_project_a import (
    _instantiate_model, _load_checkpoint, _normalize_state_dict_keys,
    _load_raw_lat_lon, compute_metrics,
)


def main():
    torch.set_num_threads(4)
    checkpoint_path = ROOT / 'model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    sd = checkpoint.get('state_dict', checkpoint)
    print('Checkpoint tensor entries:', len(sd), flush=True)
    reference = NormalizedRegressor(pretrained=False)
    print('Training architecture strict load:', reference.load_state_dict(sd, strict=True), flush=True)
    stats = compute_norm_stats(str(ROOT / 'dataset/train_split.csv'))
    reference.set_normalization(stats)
    reference.eval()
    submission = _instantiate_model(model_module, '__no_weights__.pth')
    submission = _load_checkpoint(submission, str(checkpoint_path))
    normalized_sd = _normalize_state_dict_keys(sd)
    target_sd = submission.model.state_dict()
    matched = [k for k, v in target_sd.items() if k in normalized_sd and torch.equal(v.cpu(), normalized_sd[k].cpu())]
    print(f'Submission weights identical to checkpoint: {len(matched)}/{len(target_sd)} entries', flush=True)
    print('Training normalization:', stats, flush=True)
    print('Submission normalization:', submission.normalization, flush=True)
    try:
        direct = Model(device='cpu', weights_path=checkpoint_path)
        print('Direct constructor checkpoint loading: OK', flush=True)
        del direct
    except Exception as exc:
        print('Direct constructor checkpoint loading FAILED:', str(exc)[:1200], flush=True)
    x, _ = prepare_data(str(ROOT / 'dataset/val_split.csv'))
    targets = _load_raw_lat_lon(str(ROOT / 'dataset/val_split.csv'))
    print('Validation images loaded:', len(x), flush=True)
    training_preds, submission_preds, raw_preds = [], [], []
    with torch.inference_mode():
        for start in range(0, len(x), 16):
            batch = x[start:start + 16]
            training_preds.extend(reference(batch).tolist())
            submission_preds.extend(submission.predict(batch))
            raw_preds.extend(reference.backbone(batch).tolist())
    for name, preds in [('Training inference', training_preds), ('Submission inference', submission_preds),
                        ('Constant training mean', [[stats.lat_mean, stats.lon_mean]] * len(targets))]:
        print(name, compute_metrics(preds, targets), flush=True)
    diff = torch.tensor(training_preds) - torch.tensor(submission_preds)
    print('Max coordinate discrepancy (degrees):', diff.abs().amax(dim=0).tolist(), flush=True)
    print('First 3 raw normalized predictions:', raw_preds[:3], flush=True)
    print('First 3 training predictions:', training_preds[:3], flush=True)
    print('First 3 submission predictions:', submission_preds[:3], flush=True)
    print('First 3 ground truth:', targets[:3], flush=True)


if __name__ == '__main__':
    main()
