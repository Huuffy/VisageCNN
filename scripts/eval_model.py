"""
eval_model.py — Evaluate best hybrid model on the val cache.

Outputs models/experiments/eval_results.json with:
  - overall accuracy, macro F1, weighted F1
  - per-class: precision, recall, f1, support
  - 7×7 confusion matrix as plain int list (rows=true, cols=predicted)

Usage:
    python scripts/eval_model.py
    python scripts/eval_model.py --model models/weights/hybrid_best_model.pth
    python scripts/eval_model.py --swa     # evaluate SWA model instead
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visage_er.config import Config
from visage_er.models.hybrid_model import create_hybrid_model


class ValCacheDataset(Dataset):
    """Reads pre-built .npz cache files from models/cache/val/."""

    def __init__(self):
        cache_dir = Config.MODELS_PATH / "cache" / "val"
        manifest = cache_dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Val cache not found at {cache_dir}. Run scripts/prepare_data.py first."
            )
        self.files = sorted(cache_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        coords    = torch.tensor(data["coords"],    dtype=torch.float32)
        face_crop = torch.tensor(
            data["face_crop"].astype(np.float32).transpose(2, 0, 1) / 255.0,
            dtype=torch.float32,
        )
        label = int(data["label"])
        return coords, face_crop, label


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = create_hybrid_model(pretrained_cnn=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Best model saves 'model_state_dict'; SWA saves bare state dict
    state = ckpt.get("model_state_dict", ckpt)
    # SWA models saved via torch.optim.swa_utils wrap keys under 'module.'
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    all_preds, all_targets = [], []
    use_amp = Config.MIXED_PRECISION and torch.cuda.is_available()

    with torch.no_grad():
        for coords, crops, labels in tqdm(loader, desc="Evaluating"):
            coords = coords.to(device)
            crops  = crops.to(device)
            if use_amp:
                from torch.amp import autocast
                with autocast("cuda"):
                    logits = model(coords, crops)
            else:
                logits = model(coords, crops)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    return np.array(all_targets), np.array(all_preds)


def build_results(targets, preds):
    emotions = Config.EMOTION_CLASSES
    acc      = 100.0 * accuracy_score(targets, preds)
    macro_f1 = float(f1_score(targets, preds, average="macro",    zero_division=0))
    w_f1     = float(f1_score(targets, preds, average="weighted", zero_division=0))
    f1_arr   = f1_score(targets, preds,       average=None, zero_division=0)
    prec_arr = precision_score(targets, preds, average=None, zero_division=0)
    rec_arr  = recall_score(targets, preds,    average=None, zero_division=0)
    cm       = confusion_matrix(targets, preds).tolist()

    per_class = {}
    for i, em in enumerate(emotions):
        support = int((targets == i).sum())
        per_class[em] = {
            "precision": round(float(prec_arr[i]), 4),
            "recall":    round(float(rec_arr[i]),  4),
            "f1":        round(float(f1_arr[i]),   4),
            "support":   support,
        }

    return {
        "overall_accuracy": round(acc, 2),
        "macro_f1":         round(macro_f1, 4),
        "weighted_f1":      round(w_f1, 4),
        "per_class":        per_class,
        "confusion_matrix": cm,
        "class_order":      emotions,
    }


def print_summary(results):
    print(f"\n{'='*50}")
    print(f"  Overall accuracy : {results['overall_accuracy']:.2f}%")
    print(f"  Macro F1         : {results['macro_f1']:.4f}")
    print(f"  Weighted F1      : {results['weighted_f1']:.4f}")
    print(f"\n  Per-class breakdown:")
    header = f"  {'Emotion':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}"
    print(header)
    print("  " + "-" * 44)
    for em, m in results["per_class"].items():
        print(f"  {em:<12} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>8}")
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    print(f"  Classes: {results['class_order']}")
    for row in results["confusion_matrix"]:
        print(f"  {row}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridEmotionNet on val set")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--swa",   action="store_true",    help="Evaluate SWA model")
    parser.add_argument("--out",   type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = Config.DEVICE

    if args.model:
        model_path = Path(args.model)
    elif args.swa:
        model_path = Config.MODELS_PATH / "weights" / "hybrid_swa_final.pth"
    else:
        model_path = Config.MODELS_PATH / "weights" / "hybrid_best_model.pth"

    if not model_path.exists():
        print(f"ERROR: checkpoint not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)

    dataset = ValCacheDataset()
    loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    print(f"Val samples: {len(dataset)}")

    targets, preds = run_eval(model, loader, device)
    results = build_results(targets, preds)
    print_summary(results)

    out_path = Path(args.out) if args.out else Config.MODELS_PATH / "experiments" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
