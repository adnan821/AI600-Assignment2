#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ===== Fixed settings (no CLI args) =====
ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
TEST_NPZ = ROOT / "processed_data" / "quickdraw_test.npz"
TRAIN_NPZ = ROOT / "processed_data" / "quickdraw_train.npz"
OUTPUT_FILE = ROOT / "submission.txt"

MODEL_KEYWORD = "patchmlp_v2_light"  
BATCH_SIZE = 1024
WRITE_CLASS_NAMES = False 


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_npz_images(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as d:
        keys = list(d.keys())
        for k in ["images", "x", "X", "test_images", "data"]:
            if k in d:
                arr = np.asarray(d[k])
                if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
                    break
        else:
            # fallback: first numeric array with ndim >=2
            arr = None
            for k in keys:
                cand = np.asarray(d[k])
                if cand.ndim >= 2 and np.issubdtype(cand.dtype, np.number):
                    arr = cand
                    break
            if arr is None:
                raise ValueError(f"Could not find image array in {npz_path}. Keys: {keys}")

    if arr.ndim == 3:      # [N,H,W]
        arr = arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 4:    # [N,H,W,C]
        arr = arr.reshape(arr.shape[0], -1)
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported test shape: {arr.shape}")

    return arr.astype(np.float32)


def load_class_names(npz_path: Path) -> List[str]:
    if not npz_path.exists():
        return []
    with np.load(npz_path, allow_pickle=True) as d:
        for k in ["class_names", "label_names", "classes"]:
            if k in d:
                return [str(x) for x in np.asarray(d[k]).reshape(-1).tolist()]
    return []


def compute_mean_std(train_npz: Path) -> Tuple[float, float]:
    x = load_npz_images(train_npz)
    if x.max() > 1.0:
        x = x / 255.0
    return float(x.mean()), float(x.std() + 1e-8)


def pick_best_checkpoint(checkpoint_dir: Path, keyword: str) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing directory: {checkpoint_dir}")

    all_ckpts = sorted(checkpoint_dir.glob("*.pth"))
    if not all_ckpts:
        raise FileNotFoundError(f"No .pth files found in {checkpoint_dir}")

    kw = keyword.lower().replace("-", "_")
    candidates = [p for p in all_ckpts if kw in p.stem.lower().replace("-", "_")]
    candidates = [p for p in candidates if "best" in p.stem.lower()] or candidates

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for keyword '{keyword}' in {checkpoint_dir}"
        )

    def file_score(p: Path):
        name = p.stem.lower()
        # optional score parsing from filename (e.g., acc0.91)
        m = re.search(r"(?:acc|val|score)[_\-]?([0-9]*\.?[0-9]+)", name)
        score = float(m.group(1)) if m else -1.0
        return (score, p.stat().st_mtime)

    return sorted(candidates, key=file_score, reverse=True)[0]


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.05, activation: str = "gelu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout: float = 0.05,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = MLP(num_tokens, token_mlp_dim, dropout=dropout, activation=activation)

        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = MLP(dim, channel_mlp_dim, dropout=dropout, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)       # [B, dim, tokens]
        y = self.token_mlp(y).transpose(1, 2)   # [B, tokens, dim]
        x = x + y

        y = self.channel_mlp(self.norm2(x))     # [B, tokens, dim]
        x = x + y
        return x


class PatchMLP_v2(nn.Module):
    def __init__(
        self,
        num_classes=15,
        img_size=28,
        patch_size=4,
        in_chans=1,
        embed_dim=192,
        depth=6,
        token_mlp_dim=96,
        channel_mlp_dim=384,
        dropout=0.05,
        activation="gelu",
    ):
        super().__init__()
        assert img_size % patch_size == 0, "patch_size must divide img_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        num_patches_per_side = img_size // patch_size
        self.num_tokens = num_patches_per_side * num_patches_per_side
        patch_dim = in_chans * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.patch_norm = nn.LayerNorm(embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=self.num_tokens,
                dim=embed_dim,
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _to_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            b = x.size(0)
            x = x.view(b, 1, self.img_size, self.img_size)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

        if x.size(1) != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} channels, got {x.size(1)}")
        return x

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b, (h // p) * (w // p), c * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = x.float()

        x = self._to_image(x)
        x = self._patchify(x)
        x = self.patch_embed(x)
        x = self.patch_norm(x + self.pos_embed)
        x = self.emb_dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            obj = obj["model_state_dict"]

    if not isinstance(obj, dict):
        raise ValueError("Checkpoint is not a state_dict dictionary.")

    if not all(torch.is_tensor(v) for v in obj.values()):
        raise ValueError("Checkpoint dict does not look like a valid state_dict.")

    return obj


def remap_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]

        nk = nk.replace(".token_mixer.", ".token_mlp.")
        nk = nk.replace(".channel_mixer.", ".channel_mlp.")
        nk = nk.replace(".ln1.", ".norm1.")
        nk = nk.replace(".ln2.", ".norm2.")
        nk = nk.replace(".fc1.", ".net.0.")
        nk = nk.replace(".fc2.", ".net.3.")
        out[nk] = v
    return out


def infer_config_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    if "patch_embed.weight" not in state or "pos_embed" not in state or "head.weight" not in state:
        raise ValueError("Checkpoint missing required PatchMLP_v2 keys.")

    patch_embed_w = state["patch_embed.weight"] 
    embed_dim = int(patch_embed_w.shape[0])
    patch_dim = int(patch_embed_w.shape[1])

    pos_embed = state["pos_embed"]               

    head_w = state["head.weight"]              
    num_classes = int(head_w.shape[0])

    block_ids = set()
    for k in state.keys():
        m = re.match(r"blocks\.(\d+)\.", k)
        if m:
            block_ids.add(int(m.group(1)))
    depth = (max(block_ids) + 1) if block_ids else 0
    if depth <= 0:
        raise ValueError("Could not infer model depth from checkpoint.")

    def find_hidden(prefix: str, input_dim: int) -> int:
        cands = []
        for k, v in state.items():
            if k.startswith(prefix) and k.endswith("weight") and getattr(v, "ndim", 0) == 2:
                if int(v.shape[1]) == input_dim:
                    cands.append((k, int(v.shape[0])))
        if not cands:
            raise ValueError(f"Could not infer hidden dim for {prefix}")
        cands.sort(key=lambda x: x[0])
        return cands[0][1]

    token_mlp_dim = find_hidden("blocks.0.token_mlp", num_tokens)
    channel_mlp_dim = find_hidden("blocks.0.channel_mlp", embed_dim)

    p = int(round(patch_dim ** 0.5))
    if p * p == patch_dim:
        in_chans = 1
        patch_size = p
    else:
        p3 = int(round((patch_dim / 3) ** 0.5))
        if 3 * p3 * p3 == patch_dim:
            in_chans = 3
            patch_size = p3
        else:
            raise ValueError(f"Cannot infer patch_size/in_chans from patch_dim={patch_dim}")

    side_tokens = int(round(num_tokens ** 0.5))
    if side_tokens * side_tokens != num_tokens:
        raise ValueError(f"num_tokens={num_tokens} is not a square.")
    img_size = side_tokens * patch_size

    return {
        "num_classes": num_classes,
        "img_size": img_size,
        "patch_size": patch_size,
        "in_chans": in_chans,
        "embed_dim": embed_dim,
        "depth": depth,
        "token_mlp_dim": token_mlp_dim,
        "channel_mlp_dim": channel_mlp_dim,
    }


def build_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    raw = torch.load(ckpt_path, map_location="cpu")
    state = extract_state_dict(raw)
    state = remap_state_keys(state)

    cfg = infer_config_from_state(state)
    model = PatchMLP_v2(
        num_classes=cfg["num_classes"],
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        token_mlp_dim=cfg["token_mlp_dim"],
        channel_mlp_dim=cfg["channel_mlp_dim"],
        dropout=0.05,
        activation="gelu",
    )

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"[WARN] Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys: {len(incompatible.unexpected_keys)}")

    model.to(device).eval()
    return model


def main():
    device = get_device()
    ckpt_path = pick_best_checkpoint(CHECKPOINT_DIR, MODEL_KEYWORD)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Using checkpoint: {ckpt_path.name}")

    model = build_model_from_checkpoint(ckpt_path, device)

    x_test = load_npz_images(TEST_NPZ)
    if x_test.max() > 1.0:
        x_test = x_test / 255.0

    mean, std = compute_mean_std(TRAIN_NPZ)
    x_test = (x_test - mean) / (std + 1e-8)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test.astype(np.float32))),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    preds = []
    with torch.inference_mode():
        for (xb,) in loader:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    preds = np.concatenate(preds).astype(np.int64)

    class_names = load_class_names(TRAIN_NPZ)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for p in preds.tolist():
            if WRITE_CLASS_NAMES and class_names and 0 <= p < len(class_names):
                f.write(class_names[p] + "\n")
            else:
                f.write(f"{p}\n")

    print(f"[OK] Predictions: {len(preds)}")
    print(f"[OK] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()