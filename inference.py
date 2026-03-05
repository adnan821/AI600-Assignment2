#!/usr/bin/env python3

#===================================
# USAGE
# python3 inference.py \
#   --checkpoint checkpoints/best_Champion_GELU_4L.pth \
#   --test-npz processed_data/quickdraw_test.npz \
#   --train-npz processed_data/quickdraw_train.npz \
#   --output submission.txt \
#   --output-format index
#========================================
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PA2 inference script (QuickDraw classifier)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint or TorchScript .pt")
    p.add_argument("--test-npz", type=str, required=True, help="Path to quickdraw_test.npz")
    p.add_argument("--output", type=str, default="submission.txt", help="Output prediction file")

    p.add_argument("--train-npz", type=str, default=None, help="Optional train npz for mean/std or class names")
    p.add_argument("--class-names", type=str, default=None, help="Optional .txt/.json class names file")

    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])

    p.add_argument("--mean", type=float, default=None, help="Normalization mean")
    p.add_argument("--std", type=float, default=None, help="Normalization std")

    p.add_argument("--output-format", type=str, default="index",
                   choices=["index", "name", "csv_index", "csv_name"])

    # Optional architecture overrides (for state_dict checkpoints)
    p.add_argument("--input-dim", type=int, default=None)
    p.add_argument("--hidden-dims", type=int, nargs="*", default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--activation", type=str, default=None, choices=["relu", "gelu", "tanh"])
    p.add_argument("--dropout", type=float, default=None)

    p.add_argument("--jit", action="store_true", help="Set if checkpoint is TorchScript model")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def load_npz_images(npz_path: Path) -> Tuple[np.ndarray, str]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as d:
        keys = list(d.keys())

        # Common candidate keys
        candidates = ["images", "x", "X", "test_images", "data"]
        image_key = None
        for k in candidates:
            if k in d and np.asarray(d[k]).ndim >= 2:
                image_key = k
                break

        # Fallback: largest numeric array by first dimension
        if image_key is None:
            best_n = -1
            for k in keys:
                arr = np.asarray(d[k])
                if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
                    if arr.shape[0] > best_n:
                        best_n = arr.shape[0]
                        image_key = k

        if image_key is None:
            raise ValueError(f"Could not find image array in {npz_path}. Keys: {keys}")

        x = np.asarray(d[image_key])

    if x.ndim == 3:  # [N,H,W]
        x = x.reshape(x.shape[0], -1)
    elif x.ndim == 4:  # [N,H,W,C]
        x = x.reshape(x.shape[0], -1)
    elif x.ndim != 2:
        raise ValueError(f"Unsupported image shape: {x.shape}")

    return x.astype(np.float32), image_key


def try_load_class_names(npz_path: Optional[Path]) -> Optional[List[str]]:
    if npz_path is None or not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=True) as d:
        for k in ["class_names", "label_names", "classes"]:
            if k in d:
                arr = np.asarray(d[k]).reshape(-1)
                return [str(x) for x in arr.tolist()]
    return None


def load_class_names_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"class names file not found: {path}")
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            # Accept {"0":"airplane",...} or {"classes":[...]}
            if "classes" in obj and isinstance(obj["classes"], list):
                return [str(x) for x in obj["classes"]]
            return [str(v) for _, v in sorted(obj.items(), key=lambda kv: int(kv[0]))]
        if isinstance(obj, list):
            return [str(x) for x in obj]
        raise ValueError("Unsupported JSON class name format.")
    # txt fallback: one class per line
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def compute_train_stats(train_npz: Path) -> Tuple[float, float]:
    x, _ = load_npz_images(train_npz)
    if x.max() > 1.0:
        x = x / 255.0
    mean = float(x.mean())
    std = float(x.std() + 1e-8)
    return mean, std


def activation_layer(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "gelu":
        return nn.GELU()
    if n == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class FlexibleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int,
                 activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        layers: List[nn.Module] = []
        d = self.input_dim
        for h in hidden_dims:
            h = int(h)
            layers.append(nn.Linear(d, h))
            layers.append(activation_layer(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, self.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model.", "network."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def unpack_checkpoint(obj) -> Tuple[Dict[str, torch.Tensor], Dict]:
    if not isinstance(obj, dict):
        raise ValueError("Expected dict checkpoint for state_dict mode.")

    meta = {}
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
        meta = obj.get("meta", obj.get("config", {})) or {}
    elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        state = obj["model_state_dict"]
        meta = obj.get("meta", obj.get("config", {})) or {}
    elif all(torch.is_tensor(v) for v in obj.values()):
        state = obj
    else:
        raise ValueError("Could not locate state_dict in checkpoint.")

    return clean_state_dict_keys(state), meta


def infer_mlp_arch(state: Dict[str, torch.Tensor]) -> Tuple[int, List[int], int]:
    linear = [(k, v) for k, v in state.items() if k.endswith("weight") and getattr(v, "ndim", 0) == 2]
    if not linear:
        raise ValueError("Could not infer MLP architecture from state_dict.")

    linear = sorted(linear, key=lambda kv: natural_key(kv[0]))
    input_dim = int(linear[0][1].shape[1])
    out_dims = [int(w.shape[0]) for _, w in linear]
    hidden_dims = out_dims[:-1]
    num_classes = out_dims[-1]
    return input_dim, hidden_dims, num_classes


def parse_hidden(meta_hidden) -> Optional[List[int]]:
    if meta_hidden is None:
        return None
    if isinstance(meta_hidden, (list, tuple)):
        return [int(x) for x in meta_hidden]
    if isinstance(meta_hidden, str):
        vals = [x for x in re.split(r"[,\s]+", meta_hidden.strip()) if x]
        return [int(x) for x in vals]
    return None


def build_model_from_state(
    state: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    meta: Dict
) -> Tuple[nn.Module, Dict, List[str], List[str]]:
    inf_in, inf_hidden, inf_classes = infer_mlp_arch(state)

    meta_hidden = parse_hidden(meta.get("hidden_dims"))
    input_dim = args.input_dim or int(meta.get("input_dim", inf_in))
    hidden_dims = args.hidden_dims if args.hidden_dims is not None else (meta_hidden or inf_hidden)
    num_classes = args.num_classes or int(meta.get("num_classes", inf_classes))
    activation = args.activation or str(meta.get("activation", "relu")).lower()
    dropout = args.dropout if args.dropout is not None else float(meta.get("dropout", 0.0))

    model = FlexibleMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        activation=activation,
        dropout=dropout,
    )

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    model_cfg = {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "num_classes": num_classes,
        "activation": activation,
        "dropout": dropout,
    }
    return model, model_cfg, missing, unexpected


def write_predictions(
    preds: np.ndarray,
    out_path: Path,
    out_format: str,
    class_names: Optional[List[str]] = None
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_format == "index":
        with out_path.open("w", encoding="utf-8") as f:
            for p in preds.tolist():
                f.write(f"{int(p)}\n")
        return

    if out_format == "name":
        if class_names is None:
            raise ValueError("class names required for output-format=name")
        with out_path.open("w", encoding="utf-8") as f:
            for p in preds.tolist():
                f.write(f"{class_names[int(p)]}\n")
        return

    if out_format in ("csv_index", "csv_name"):
        use_names = out_format == "csv_name"
        if use_names and class_names is None:
            raise ValueError("class names required for output-format=csv_name")
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prediction"])
            for i, p in enumerate(preds.tolist()):
                pred_val = class_names[int(p)] if use_names else int(p)
                writer.writerow([i, pred_val])
        return

    raise ValueError(f"Unsupported output format: {out_format}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_npz = Path(args.test_npz)
    train_npz = Path(args.train_npz) if args.train_npz else None

    # Model loading
    meta: Dict = {}
    if args.jit:
        model = torch.jit.load(str(ckpt_path), map_location=device)
        model_cfg = {}
        missing, unexpected = [], []
    else:
        ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
        state, meta = unpack_checkpoint(ckpt_obj)
        model, model_cfg, missing, unexpected = build_model_from_state(state, args, meta)

    model.to(device)
    model.eval()

    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    # Data loading
    x_test, used_key = load_npz_images(test_npz)
    if x_test.max() > 1.0:
        x_test = x_test / 255.0

    # Normalization
    mean = args.mean
    std = args.std

    if mean is None and "mean" in meta:
        mean = float(meta["mean"])
    if std is None and "std" in meta:
        std = float(meta["std"])

    if mean is None or std is None:
        if train_npz is not None and train_npz.exists():
            mean, std = compute_train_stats(train_npz)
            print(f"[INFO] Computed train stats: mean={mean:.6f}, std={std:.6f}")
        else:
            mean, std = 0.0, 1.0
            print("[WARN] mean/std not provided or found in checkpoint. Using identity normalization.")

    x_test = (x_test - float(mean)) / (float(std) + 1e-8)

    expected_dim = getattr(model, "input_dim", None)
    if expected_dim is not None and x_test.shape[1] != int(expected_dim):
        raise ValueError(
            f"Input feature mismatch: test has {x_test.shape[1]} features, model expects {expected_dim}."
        )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    preds = []
    with torch.inference_mode():
        for (xb,) in loader:
            logits = model(xb.to(device))
            p = torch.argmax(logits, dim=1)
            preds.append(p.cpu().numpy())

    preds = np.concatenate(preds).astype(np.int64)

    # Class names if needed
    class_names = None
    if args.output_format in ("name", "csv_name"):
        if args.class_names:
            class_names = load_class_names_file(Path(args.class_names))
        if class_names is None:
            class_names = try_load_class_names(test_npz)
        if class_names is None and train_npz is not None:
            class_names = try_load_class_names(train_npz)
        if class_names is None:
            raise ValueError("Could not find class names. Provide --class-names.")
        if max(preds.tolist()) >= len(class_names):
            raise ValueError("Predicted class index exceeds class names length.")

    out_path = Path(args.output)
    write_predictions(preds, out_path, args.output_format, class_names)

    print(f"[OK] Device: {device}")
    print(f"[OK] Loaded test key: {used_key}")
    print(f"[OK] Total predictions: {len(preds)}")
    print(f"[OK] Output written to: {out_path}")


if __name__ == "__main__":
    main()