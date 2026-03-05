#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Fixed config (no CLI args)
# =========================
ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
TEST_NPZ = ROOT / "processed_data" / "quickdraw_test.npz"
TRAIN_NPZ = ROOT / "processed_data" / "quickdraw_train.npz"
OUTPUT_TXT = ROOT / "submission.txt"

TARGET_MODEL_KEY = "patchmlp_v2_light"
BATCH_SIZE = 1024


def choose_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _first_numeric_array(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        for k in ("images", "x", "X", "test_images", "data"):
            if k in data:
                arr = np.asarray(data[k])
                if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
                    return arr

        for k in data.files:
            arr = np.asarray(data[k])
            if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
                return arr

    raise ValueError(f"No usable image array found in: {npz_path}")


def load_flat_images(npz_path: Path) -> np.ndarray:
    x = _first_numeric_array(npz_path)

    if x.ndim == 2:
        pass
    elif x.ndim == 3:       # [N,H,W]
        x = x.reshape(x.shape[0], -1)
    elif x.ndim == 4:       # [N,H,W,C]
        x = x.reshape(x.shape[0], -1)
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")

    return x.astype(np.float32)


def compute_train_stats(train_npz: Path) -> Tuple[float, float]:
    if not train_npz.exists():
        return 0.0, 1.0
    try:
        x = load_flat_images(train_npz)
        if x.max() > 1.0:
            x = x / 255.0
        return float(x.mean()), float(x.std() + 1e-8)
    except Exception:
        return 0.0, 1.0


def pick_best_light_checkpoint(ckpt_dir: Path) -> Path:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    all_ckpts = sorted(ckpt_dir.glob("*.pth"))
    if not all_ckpts:
        raise FileNotFoundError(f"No .pth checkpoints in: {ckpt_dir}")

    def norm(s: str) -> str:
        return s.lower().replace("-", "_")

    light = [p for p in all_ckpts if TARGET_MODEL_KEY in norm(p.stem)]
    if not light:
        available = "\n".join(f" - {p.name}" for p in all_ckpts)
        raise FileNotFoundError(
            f"No checkpoint matching '{TARGET_MODEL_KEY}'. Available:\n{available}"
        )

    best_named = [p for p in light if "best" in p.stem.lower()]
    candidates = best_named if best_named else light
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def activation_layer(name: str) -> nn.Module:
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
            activation_layer(activation),
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
        y = self.norm1(x).transpose(1, 2)        # [B, D, T]
        y = self.token_mlp(y).transpose(1, 2)    # [B, T, D]
        x = x + y

        y = self.channel_mlp(self.norm2(x))      # [B, T, D]
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

        side = img_size // patch_size
        self.num_tokens = side * side
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
            raise ValueError(f"Expected {self.in_chans} channel(s), got {x.size(1)}")
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


def extract_state_dict_or_model(obj) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[nn.Module]]:
    if isinstance(obj, nn.Module):
        return None, obj

    # TorchScript model
    try:
        if isinstance(obj, torch.jit.ScriptModule):
            return None, obj
    except Exception:
        pass

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            maybe = obj.get(key, None)
            if isinstance(maybe, dict):
                sd = {k: v for k, v in maybe.items() if torch.is_tensor(v)}
                if sd:
                    return sd, None

        sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if sd:
            return sd, None

    raise ValueError("Unsupported checkpoint format.")


def _strip_prefixes(k: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in ("module.", "model.", "network.", "backbone."):
            if k.startswith(p):
                k = k[len(p):]
                changed = True
    return k


def normalize_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remap = [
        (".token_mixer.", ".token_mlp."),
        (".channel_mixer.", ".channel_mlp."),
        (".mlp_tokens.", ".token_mlp."),
        (".mlp_channels.", ".channel_mlp."),
        (".ln1.", ".norm1."),
        (".ln2.", ".norm2."),
        (".norm_1.", ".norm1."),
        (".norm_2.", ".norm2."),
        (".fc1.", ".net.0."),
        (".fc2.", ".net.3."),
        (".linear1.", ".net.0."),
        (".linear2.", ".net.3."),
    ]

    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = _strip_prefixes(k)
        for a, b in remap:
            nk = nk.replace(a, b)
        out[nk] = v
    return out


def _key_by_suffix(state: Dict[str, torch.Tensor], suffixes) -> Optional[str]:
    for s in suffixes:
        if s in state:
            return s
    for k in state.keys():
        for s in suffixes:
            if k.endswith(s):
                return k
    return None


def infer_patchmlp_config(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    k_patch = _key_by_suffix(state, ["patch_embed.weight"])
    k_pos = _key_by_suffix(state, ["pos_embed"])
    k_head = _key_by_suffix(state, ["head.weight"])

    if k_patch is None or k_pos is None or k_head is None:
        raise ValueError("Could not find patch_embed/pos_embed/head in checkpoint.")

    patch_w = state[k_patch]   # [embed_dim, patch_dim]
    pos = state[k_pos]         # [1, num_tokens, embed_dim]
    head_w = state[k_head]     # [num_classes, embed_dim]

    if patch_w.ndim != 2 or pos.ndim != 3 or head_w.ndim != 2:
        raise ValueError("Invalid checkpoint tensor shapes for PatchMLP_v2.")

    embed_dim = int(patch_w.shape[0])
    patch_dim = int(patch_w.shape[1])
    num_tokens = int(pos.shape[1])
    num_classes = int(head_w.shape[0])

    block_ids = []
    for k in state.keys():
        m = re.search(r"(?:blocks|mixer_blocks|layers)\.(\d+)\.", k)
        if m:
            block_ids.append(int(m.group(1)))
    if not block_ids:
        raise ValueError("Could not infer depth (no block keys found).")
    depth = max(block_ids) + 1

    def infer_hidden(in_dim: int, marks) -> int:
        for mark in marks:
            for k, v in state.items():
                if mark in k and k.endswith("weight") and getattr(v, "ndim", 0) == 2:
                    if int(v.shape[1]) == in_dim:
                        return int(v.shape[0])

        # fallback: first block with matching input dim
        for k, v in state.items():
            if (("blocks.0" in k) or ("layers.0" in k)) and k.endswith("weight") and getattr(v, "ndim", 0) == 2:
                if int(v.shape[1]) == in_dim:
                    return int(v.shape[0])

        raise ValueError(f"Could not infer hidden dim for in_dim={in_dim}")

    token_mlp_dim = infer_hidden(
        num_tokens,
        marks=[
            "blocks.0.token_mlp.net.0",
            "blocks.0.token_mlp.fc1",
            "layers.0.token_mlp.net.0",
            "layers.0.token_mlp.fc1",
        ],
    )
    channel_mlp_dim = infer_hidden(
        embed_dim,
        marks=[
            "blocks.0.channel_mlp.net.0",
            "blocks.0.channel_mlp.fc1",
            "layers.0.channel_mlp.net.0",
            "layers.0.channel_mlp.fc1",
        ],
    )

    p = int(round(patch_dim ** 0.5))
    if p * p == patch_dim:
        in_chans = 1
        patch_size = p
    else:
        p3 = int(round((patch_dim / 3.0) ** 0.5))
        if 3 * p3 * p3 != patch_dim:
            raise ValueError(f"Cannot infer patch_size from patch_dim={patch_dim}")
        in_chans = 3
        patch_size = p3

    side_tokens = int(round(num_tokens ** 0.5))
    if side_tokens * side_tokens != num_tokens:
        raise ValueError("num_tokens is not a square; cannot infer img_size.")
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
    state, model_obj = extract_state_dict_or_model(raw)

    if model_obj is not None:
        return model_obj.to(device).eval()

    assert state is not None
    state = normalize_state_keys(state)
    cfg = infer_patchmlp_config(state)

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

    model_sd = model.state_dict()
    matched = sum(1 for k, v in state.items() if k in model_sd and model_sd[k].shape == v.shape)
    if matched < max(10, int(0.5 * len(model_sd))):
        raise RuntimeError(f"Checkpoint mismatch: matched {matched}/{len(model_sd)} keys.")

    incompat = model.load_state_dict(state, strict=False)
    if incompat.missing_keys:
        print(f"[WARN] Missing keys: {len(incompat.missing_keys)}")
    if incompat.unexpected_keys:
        print(f"[WARN] Unexpected keys: {len(incompat.unexpected_keys)}")

    return model.to(device).eval()


def predict(model: nn.Module, x_test: np.ndarray, device: torch.device) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x_test.astype(np.float32)))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds = []
    with torch.inference_mode():
        for (xb,) in dl:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())

    return np.concatenate(preds).astype(np.int64)


def main() -> None:
    device = choose_device()
    ckpt = pick_best_light_checkpoint(CHECKPOINT_DIR)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {ckpt.name}")

    model = build_model_from_checkpoint(ckpt, device)

    x_test = load_flat_images(TEST_NPZ)
    if x_test.max() > 1.0:
        x_test = x_test / 255.0

    mean, std = compute_train_stats(TRAIN_NPZ)
    x_test = (x_test - mean) / (std + 1e-8)

    preds = predict(model, x_test, device)
    np.savetxt(OUTPUT_TXT, preds, fmt="%d")

    print(f"[OK] Predictions: {len(preds)}")
    print(f"[OK] Output: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()