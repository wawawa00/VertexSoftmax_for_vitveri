#!/usr/bin/env python3
"""Tiny ViT-block benchmark for Vertex-CROWN certification.

This is the next-step benchmark after ``tiny_vit_vertex_benchmark.py``.  The
old script certifies a single-head attention classifier with no residual path.
This script moves closer to a ViT block while keeping the vertex certificate
mathematically sound:

    patches -> linear embedding + position
            -> multi-head self-attention + output projection + residual
            -> mean pool -> classifier

For this attention-residual block, the class margin is still linear downstream
of the attention output, so the score-box vertex-softmax primitive applies
exactly row-by-row.  The optional ``full_mlp`` block mode appends a ReLU MLP
residual.  For that mode, ``objective_vertex_crown`` first lower-bounds the
nonlinear suffix by a CROWN-style affine function of the attention-residual
state, then passes the resulting tokenwise objective coefficients to the exact
score-box primitive.  ``vertex_crown`` is kept as a backwards-compatible alias.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import pickle
import struct
import sys
import tarfile
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "alpha-beta-CROWN"))
sys.path.insert(0, str(ROOT / "third_party" / "alpha-beta-CROWN" / "auto_LiRPA"))

try:
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
except Exception as exc:  # pragma: no cover - reported in main.
    BoundedModule = None
    BoundedTensor = None
    PerturbationLpNorm = None
    AUTOLIRPA_IMPORT_ERROR = exc
else:
    AUTOLIRPA_IMPORT_ERROR = None


DATASET_URLS = {
    "mnist": {
        "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    },
    "fashion_mnist": {
        "train_images": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
        "train_labels": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
        "test_images": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
        "test_labels": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
    },
}

CIFAR10_URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://mirrors.dotsrc.org/osdn/datasets/74526/cifar-10-python.tar.gz",
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
]
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_ROOT = "cifar-10-batches-py"


def dataset_choices() -> list[str]:
    return sorted([*DATASET_URLS.keys(), "cifar10", "cifar10_gray"])


@dataclass
class SummaryRow:
    dataset: str
    classes: str
    seed: int
    block_mode: str
    norm_mode: str
    layernorm_eps: float
    num_blocks: int
    patch_size: int
    dim: int
    heads: int
    mlp_dim: int
    epsilon: float
    method: str
    clean_acc: float
    pgd_acc: float
    cert_acc: float
    mean_lower: float
    median_lower: float
    images: int
    elapsed_sec: float
    sec_per_image: float
    errors: int


@dataclass
class DetailRow:
    dataset: str
    classes: str
    seed: int
    block_mode: str
    norm_mode: str
    layernorm_eps: float
    num_blocks: int
    patch_size: int
    dim: int
    heads: int
    mlp_dim: int
    epsilon: float
    method: str
    index: int
    label: int
    pred: int
    pgd_margin: float
    lower_bound: float
    certified: int
    elapsed_sec: float
    error: str


@dataclass
class TargetDetailRow:
    dataset: str
    classes: str
    seed: int
    block_mode: str
    norm_mode: str
    layernorm_eps: float
    num_blocks: int
    patch_size: int
    dim: int
    heads: int
    mlp_dim: int
    epsilon: float
    index: int
    label: int
    pred: int
    target: int
    pgd_margin: float
    crown_lower: float
    objective_vertex_lower: float
    hybrid_lower: float
    winner: str
    crown_worst_target: int
    objective_vertex_worst_target: int
    hybrid_worst_target: int
    crown_certified_target: int
    objective_vertex_certified_target: int
    hybrid_certified_target: int
    elapsed_sec: float
    error: str


class TinyViTBlock(torch.nn.Module):
    """Small ViT-style block with manually written attention for auto_LiRPA."""

    def __init__(
        self,
        patch_dim: int,
        dim: int,
        heads: int,
        num_classes: int,
        seq_len: int,
        block_mode: str,
        norm_mode: str,
        layernorm_eps: float,
        mlp_dim: int,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if block_mode not in {"attention_residual", "full_mlp"}:
            raise ValueError(f"unknown block_mode {block_mode}")
        if norm_mode not in {"none", "pre_layernorm"}:
            raise ValueError(f"unknown norm_mode {norm_mode}")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.seq_len = seq_len
        self.block_mode = block_mode
        self.norm_mode = norm_mode
        self.mlp_dim = mlp_dim

        self.pos = torch.nn.Parameter(torch.zeros(seq_len, dim))
        self.embed = torch.nn.Linear(patch_dim, dim)
        self.layernorm_eps = layernorm_eps
        self.norm1 = torch.nn.LayerNorm(dim, eps=layernorm_eps) if norm_mode == "pre_layernorm" else torch.nn.Identity()
        self.norm2 = torch.nn.LayerNorm(dim, eps=layernorm_eps) if norm_mode == "pre_layernorm" else torch.nn.Identity()
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim)
        self.out = torch.nn.Linear(dim, dim)
        if block_mode == "full_mlp":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, mlp_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mlp_dim, dim),
            )
        else:
            self.mlp = torch.nn.Identity()
        self.cls = torch.nn.Linear(dim, num_classes)
        torch.nn.init.normal_(self.pos, std=0.02)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) + self.pos.unsqueeze(0)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _dim = x.shape
        return x.reshape(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

    def attention_input(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm1(z)

    def attention_scores(self, z: torch.Tensor) -> torch.Tensor:
        attn_in = self.attention_input(z)
        q = self.split_heads(self.q(attn_in))
        k = self.split_heads(self.k(attn_in))
        return q.matmul(k.transpose(-1, -2)) / math.sqrt(self.head_dim)

    def attention_context(self, z: torch.Tensor) -> torch.Tensor:
        attn_in = self.attention_input(z)
        v = self.split_heads(self.v(attn_in))
        scores = self.attention_scores(z)
        scores = scores - scores.amax(dim=-1, keepdim=True)
        exp_scores = torch.exp(scores)
        attn = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
        ctx = attn.matmul(v)
        return ctx.transpose(1, 2).reshape(z.shape[0], z.shape[1], self.dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.encode(tokens)
        h = z + self.out(self.attention_context(z))
        if self.block_mode == "full_mlp":
            h = h + self.mlp(self.norm2(h))
        return self.cls(h.mean(dim=1))


class TinyTransformerLayer(torch.nn.Module):
    """One residual self-attention + optional ReLU-MLP transformer-style layer."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, block_mode: str, norm_mode: str, layernorm_eps: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if block_mode not in {"attention_residual", "full_mlp"}:
            raise ValueError(f"unknown block_mode {block_mode}")
        if norm_mode not in {"none", "pre_layernorm"}:
            raise ValueError(f"unknown norm_mode {norm_mode}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.block_mode = block_mode
        self.norm_mode = norm_mode
        self.layernorm_eps = layernorm_eps
        self.norm1 = torch.nn.LayerNorm(dim, eps=layernorm_eps) if norm_mode == "pre_layernorm" else torch.nn.Identity()
        self.norm2 = torch.nn.LayerNorm(dim, eps=layernorm_eps) if norm_mode == "pre_layernorm" else torch.nn.Identity()
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim)
        self.out = torch.nn.Linear(dim, dim)
        if block_mode == "full_mlp":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, mlp_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mlp_dim, dim),
            )
        else:
            self.mlp = torch.nn.Identity()

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _dim = x.shape
        return x.reshape(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

    def attention_input(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm1(z)

    def attention_scores(self, z: torch.Tensor) -> torch.Tensor:
        attn_in = self.attention_input(z)
        q = self.split_heads(self.q(attn_in))
        k = self.split_heads(self.k(attn_in))
        return q.matmul(k.transpose(-1, -2)) / math.sqrt(self.head_dim)

    def attention_context(self, z: torch.Tensor) -> torch.Tensor:
        attn_in = self.attention_input(z)
        v = self.split_heads(self.v(attn_in))
        scores = self.attention_scores(z)
        scores = scores - scores.amax(dim=-1, keepdim=True)
        exp_scores = torch.exp(scores)
        attn = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
        ctx = attn.matmul(v)
        return ctx.transpose(1, 2).reshape(z.shape[0], z.shape[1], self.dim)

    def attention_residual(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.out(self.attention_context(z))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.attention_residual(z)
        if self.block_mode == "full_mlp":
            h = h + self.mlp(self.norm2(h))
        return h


class TinyMultiBlockViT(torch.nn.Module):
    """Small multi-block ViT-style classifier used for the real-transformer track."""

    def __init__(
        self,
        patch_dim: int,
        dim: int,
        heads: int,
        num_classes: int,
        seq_len: int,
        block_mode: str,
        norm_mode: str,
        layernorm_eps: float,
        mlp_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.seq_len = seq_len
        self.block_mode = block_mode
        self.norm_mode = norm_mode
        self.layernorm_eps = layernorm_eps
        self.mlp_dim = mlp_dim
        self.num_blocks = num_blocks
        self.pos = torch.nn.Parameter(torch.zeros(seq_len, dim))
        self.embed = torch.nn.Linear(patch_dim, dim)
        self.layers = torch.nn.ModuleList(
            [TinyTransformerLayer(dim, heads, mlp_dim, block_mode, norm_mode, layernorm_eps) for _ in range(num_blocks)]
        )
        self.cls = torch.nn.Linear(dim, num_classes)
        torch.nn.init.normal_(self.pos, std=0.02)

    @property
    def final_layer(self) -> TinyTransformerLayer:
        return self.layers[-1]

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) + self.pos.unsqueeze(0)

    def final_block_input(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.encode(tokens)
        for layer in self.layers[:-1]:
            z = layer(z)
        return z

    def final_attention_scores(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.final_layer.attention_scores(self.final_block_input(tokens))

    def final_attention_input(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.final_layer.attention_input(self.final_block_input(tokens))

    def final_attention_residual(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.final_layer.attention_residual(self.final_block_input(tokens))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.encode(tokens)
        for layer in self.layers:
            z = layer(z)
        return self.cls(z.mean(dim=1))


class FinalBlockAdapter:
    """Attribute adapter exposing the final block through the one-block helpers."""

    def __init__(self, model: TinyMultiBlockViT):
        self.dim = model.dim
        self.heads = model.heads
        self.head_dim = model.head_dim
        self.seq_len = model.seq_len
        self.block_mode = model.block_mode
        self.norm_mode = model.norm_mode
        self.mlp_dim = model.mlp_dim
        self.q = model.final_layer.q
        self.k = model.final_layer.k
        self.v = model.final_layer.v
        self.out = model.final_layer.out
        self.mlp = model.final_layer.mlp
        self.cls = model.cls


class MultiheadAttentionScoreModule(torch.nn.Module):
    def __init__(self, model: TinyViTBlock):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.model.encode(tokens)
        return self.model.attention_scores(z)


class AttentionResidualModule(torch.nn.Module):
    def __init__(self, model: TinyViTBlock):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.model.encode(tokens)
        return z + self.model.out(self.model.attention_context(z))


class AttentionInputModule(torch.nn.Module):
    def __init__(self, model: TinyViTBlock):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.attention_input(self.model.encode(tokens))


class FinalBlockInputModule(torch.nn.Module):
    def __init__(self, model: TinyMultiBlockViT):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.final_block_input(tokens)


class FinalBlockScoreModule(torch.nn.Module):
    def __init__(self, model: TinyMultiBlockViT):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.final_attention_scores(tokens)


class FinalBlockAttentionInputModule(torch.nn.Module):
    def __init__(self, model: TinyMultiBlockViT):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.final_attention_input(tokens)


class FinalBlockAttentionResidualModule(torch.nn.Module):
    def __init__(self, model: TinyMultiBlockViT):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.final_attention_residual(tokens)


class MarginModule(torch.nn.Module):
    def __init__(self, model: TinyViTBlock, label: int):
        super().__init__()
        self.model = model
        self.label = int(label)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.model(tokens)
        true_logit = logits[:, self.label : self.label + 1]
        return true_logit - logits


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path)


def download_any(path: Path, urls: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    errors = []
    for url in urls:
        try:
            download(path, url)
            if path.exists() and path.stat().st_size > 0:
                return
        except Exception as exc:
            errors.append(f"{url}: {exc!r}")
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    raise RuntimeError("all download attempts failed:\n" + "\n".join(errors))


def read_idx_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic {magic} in {path}")
        data = torch.frombuffer(f.read(), dtype=torch.uint8).clone()
    return data.reshape(count, 1, rows, cols).float() / 255.0


def read_idx_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic {magic} in {path}")
        data = torch.frombuffer(f.read(), dtype=torch.uint8).clone()
    return data.reshape(count).long()


def load_idx_dataset(
    data_dir: Path,
    dataset: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset not in DATASET_URLS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(DATASET_URLS)}")
    urls = DATASET_URLS[dataset]
    paths = {name: data_dir / dataset / url.rsplit("/", 1)[-1] for name, url in urls.items()}
    for name, url in urls.items():
        download(paths[name], url)
    return (
        read_idx_images(paths["train_images"]),
        read_idx_labels(paths["train_labels"]),
        read_idx_images(paths["test_images"]),
        read_idx_labels(paths["test_labels"]),
    )


def read_cifar_batch(
    archive: tarfile.TarFile,
    batch_name: str,
    grayscale: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    member = archive.getmember(f"{CIFAR10_ROOT}/{batch_name}")
    with archive.extractfile(member) as handle:
        if handle is None:
            raise FileNotFoundError(f"missing {batch_name} in CIFAR-10 archive")
        payload = pickle.load(handle, encoding="latin1")
    data = torch.tensor(payload["data"], dtype=torch.uint8).reshape(-1, 3, 32, 32).float() / 255.0
    if grayscale:
        data = (
            0.2989 * data[:, 0:1]
            + 0.5870 * data[:, 1:2]
            + 0.1140 * data[:, 2:3]
        )
    labels = torch.tensor(payload["labels"], dtype=torch.long)
    return data.contiguous(), labels


def load_cifar10(data_dir: Path, grayscale: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset_name = "cifar10_gray" if grayscale else "cifar10"
    cifar_dir = data_dir / dataset_name
    archive_path = cifar_dir / CIFAR10_ARCHIVE
    download_any(archive_path, CIFAR10_URLS)
    train_images = []
    train_labels = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for batch_idx in range(1, 6):
            images, labels = read_cifar_batch(archive, f"data_batch_{batch_idx}", grayscale)
            train_images.append(images)
            train_labels.append(labels)
        test_images, test_labels = read_cifar_batch(archive, "test_batch", grayscale)
    return (
        torch.cat(train_images, dim=0),
        torch.cat(train_labels, dim=0),
        test_images,
        test_labels,
    )


def load_dataset(
    data_dir: Path,
    dataset: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset == "cifar10":
        return load_cifar10(data_dir, grayscale=False)
    if dataset == "cifar10_gray":
        return load_cifar10(data_dir, grayscale=True)
    return load_idx_dataset(data_dir, dataset)


def select_classes(
    images: torch.Tensor,
    labels: torch.Tensor,
    classes: list[int],
    limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for cls in classes:
        mask |= labels == cls
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if limit is not None:
        idx = idx[:limit]
    remap = {cls: i for i, cls in enumerate(classes)}
    selected_labels = torch.tensor([remap[int(labels[i])] for i in idx], dtype=torch.long)
    return images[idx], selected_labels


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).contiguous()


def train_model(
    model: TinyViTBlock,
    train_tokens: torch.Tensor,
    train_labels: torch.Tensor,
    test_tokens: torch.Tensor,
    test_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = train_tokens.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        correct = 0
        seen = 0
        model.train()
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            xb = train_tokens[idx].to(device)
            yb = train_labels[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += float(loss.detach().cpu()) * xb.shape[0]
            correct += (logits.argmax(dim=1) == yb).sum().item()
            seen += xb.shape[0]
        clean = evaluate_clean(model, test_tokens, test_labels, args.eval_limit, device)
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={total_loss / seen:.4f} "
            f"train_acc={correct / seen:.4f} eval_acc={clean:.4f}",
            flush=True,
        )
    return evaluate_clean(model, test_tokens, test_labels, None, device)


@torch.no_grad()
def evaluate_clean(
    model: TinyViTBlock,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    limit: int | None,
    device: torch.device,
) -> float:
    model.eval()
    n = tokens.shape[0] if limit is None else min(limit, tokens.shape[0])
    correct = 0
    seen = 0
    for start in range(0, n, 512):
        xb = tokens[start : start + 512].to(device)
        yb = labels[start : start + 512].to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        seen += xb.shape[0]
    return correct / max(1, seen)


def pgd_worst_margin(
    model: TinyViTBlock,
    image: torch.Tensor,
    label: int,
    eps: float,
    patch_size: int,
    steps: int,
    restarts: int,
    step_size: float,
    device: torch.device,
) -> float:
    model.eval()
    image = image.to(device)
    best = float("inf")
    for _ in range(restarts):
        delta = torch.empty_like(image).uniform_(-eps, eps)
        delta = torch.clamp(image + delta, 0.0, 1.0) - image
        for _step in range(steps):
            delta.requires_grad_(True)
            tokens = patchify((image + delta).unsqueeze(0), patch_size)
            logits = model(tokens)
            target_logits = logits.clone()
            target_logits[:, label] = -1e9
            worst_target = target_logits.argmax(dim=1)
            margin = logits[:, label] - logits.gather(1, worst_target[:, None]).squeeze(1)
            grad = torch.autograd.grad(margin.sum(), delta)[0]
            delta = (delta - step_size * grad.sign()).detach()
            delta = torch.clamp(delta, -eps, eps)
            delta = torch.clamp(image + delta, 0.0, 1.0) - image
        with torch.no_grad():
            tokens = patchify((image + delta).unsqueeze(0), patch_size)
            logits = model(tokens)
            target_logits = logits.clone()
            target_logits[:, label] = -1e9
            worst_target = target_logits.argmax(dim=1)
            margin = logits[:, label] - logits.gather(1, worst_target[:, None]).squeeze(1)
            best = min(best, float(margin.detach().cpu()))
    return best


def softmax_box_expectation_min(score_l: torch.Tensor, score_u: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """Exact min of softmax(score)^T coeff over an independent score box.

    score_l, score_u: B x Q x K.
    coeff: B x K, shared across query rows, or B x Q x K.
    Returns B x Q.
    """
    batch, queries, keys = score_l.shape
    shift = score_u.amax(dim=-1, keepdim=True)
    y_l = torch.exp(score_l - shift)
    y_u = torch.exp(score_u - shift)
    if coeff.dim() == 2:
        order = coeff.argsort(dim=1)
        order_q = order[:, None, :].expand(batch, queries, keys)
        c_sorted = coeff.gather(dim=1, index=order)[:, None, :].expand(batch, queries, keys)
    elif coeff.dim() == 3:
        order_q = coeff.argsort(dim=2)
        c_sorted = coeff.gather(dim=2, index=order_q)
    else:
        raise ValueError(f"coeff must have shape BxK or BxQxK, got {tuple(coeff.shape)}")
    y_l_sorted = y_l.gather(dim=2, index=order_q)
    y_u_sorted = y_u.gather(dim=2, index=order_q)
    zero = torch.zeros(batch, queries, 1, device=score_l.device, dtype=score_l.dtype)
    prefix_num_u = torch.cat((zero, (c_sorted * y_u_sorted).cumsum(dim=2)), dim=2)
    prefix_den_u = torch.cat((zero, y_u_sorted.cumsum(dim=2)), dim=2)
    prefix_num_l = torch.cat((zero, (c_sorted * y_l_sorted).cumsum(dim=2)), dim=2)
    prefix_den_l = torch.cat((zero, y_l_sorted.cumsum(dim=2)), dim=2)
    total_num_l = prefix_num_l[:, :, -1:]
    total_den_l = prefix_den_l[:, :, -1:]
    suffix_num_l = total_num_l - prefix_num_l
    suffix_den_l = total_den_l - prefix_den_l
    values = (prefix_num_u + suffix_num_l) / (prefix_den_u + suffix_den_l)
    return values.amin(dim=2)


def affine_lower(coeff: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return bias + torch.where(coeff >= 0, coeff * lower, coeff * upper).sum(dim=-1)


def crown_margin_lower(
    model: TinyViTBlock,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    method: str,
    alpha_iters: int,
) -> torch.Tensor:
    if BoundedModule is None:
        raise RuntimeError(f"auto_LiRPA import failed: {AUTOLIRPA_IMPORT_ERROR!r}")
    margin_model = MarginModule(model, label).eval()
    bounded = BoundedModule(
        margin_model,
        (center,),
        bound_opts={
            "verbosity": 0,
            "fixed_reducemax_index": True,
            "optimize_bound_args": {
                "iteration": alpha_iters,
                "lr_alpha": 0.1,
            },
        },
    )
    bounded_x = BoundedTensor(center, PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper))
    lb, _ = bounded.compute_bounds(
        x=(bounded_x,),
        method=method,
        bound_lower=True,
        bound_upper=False,
    )
    return lb.reshape(-1)


def crown_score_bounds(
    model: TinyViTBlock,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if BoundedModule is None:
        raise RuntimeError(f"auto_LiRPA import failed: {AUTOLIRPA_IMPORT_ERROR!r}")
    score_model = MultiheadAttentionScoreModule(model).eval()
    bounded = BoundedModule(score_model, (center,), bound_opts={"verbosity": 0})
    bounded_x = BoundedTensor(center, PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper))
    score_l, score_u = bounded.compute_bounds(
        x=(bounded_x,),
        method="CROWN",
        bound_lower=True,
        bound_upper=True,
    )
    return score_l, score_u


def crown_attention_residual_bounds(
    model: TinyViTBlock,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if BoundedModule is None:
        raise RuntimeError(f"auto_LiRPA import failed: {AUTOLIRPA_IMPORT_ERROR!r}")
    h_model = AttentionResidualModule(model).eval()
    bounded = BoundedModule(
        h_model,
        (center,),
        bound_opts={"verbosity": 0, "fixed_reducemax_index": True},
    )
    bounded_x = BoundedTensor(center, PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper))
    h_l, h_u = bounded.compute_bounds(
        x=(bounded_x,),
        method="CROWN",
        bound_lower=True,
        bound_upper=True,
    )
    return h_l, h_u


def crown_attention_input_bounds(
    model: TinyViTBlock,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if BoundedModule is None:
        raise RuntimeError(f"auto_LiRPA import failed: {AUTOLIRPA_IMPORT_ERROR!r}")
    input_model = AttentionInputModule(model).eval()
    bounded = BoundedModule(
        input_model,
        (center,),
        bound_opts={"verbosity": 0, "fixed_reducemax_index": True},
    )
    bounded_x = BoundedTensor(center, PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper))
    z_l, z_u = bounded.compute_bounds(
        x=(bounded_x,),
        method="CROWN",
        bound_lower=True,
        bound_upper=True,
    )
    return z_l, z_u


def crown_final_block_input_bounds(
    model: TinyMultiBlockViT,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if BoundedModule is None:
        raise RuntimeError(f"auto_LiRPA import failed: {AUTOLIRPA_IMPORT_ERROR!r}")
    prefix_model = FinalBlockInputModule(model).eval()
    bounded = BoundedModule(prefix_model, (center,), bound_opts={"verbosity": 0, "fixed_reducemax_index": True})
    bounded_x = BoundedTensor(center, PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper))
    z_l, z_u = bounded.compute_bounds(
        x=(bounded_x,),
        method="CROWN",
        bound_lower=True,
        bound_upper=True,
    )
    return z_l, z_u


def crown_final_score_bounds(
    model: TinyMultiBlockViT,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if BoundedModule is None:
        raise RuntimeError(f"auto_LiRPA import failed: {AUTOLIRPA_IMPORT_ERROR!r}")
    score_model = FinalBlockScoreModule(model).eval()
    bounded = BoundedModule(score_model, (center,), bound_opts={"verbosity": 0, "fixed_reducemax_index": True})
    bounded_x = BoundedTensor(center, PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper))
    score_l, score_u = bounded.compute_bounds(
        x=(bounded_x,),
        method="CROWN",
        bound_lower=True,
        bound_upper=True,
    )
    return score_l, score_u


def crown_final_attention_input_bounds(
    model: TinyMultiBlockViT,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if BoundedModule is None:
        raise RuntimeError(f"auto_LiRPA import failed: {AUTOLIRPA_IMPORT_ERROR!r}")
    input_model = FinalBlockAttentionInputModule(model).eval()
    bounded = BoundedModule(input_model, (center,), bound_opts={"verbosity": 0, "fixed_reducemax_index": True})
    bounded_x = BoundedTensor(center, PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper))
    z_l, z_u = bounded.compute_bounds(
        x=(bounded_x,),
        method="CROWN",
        bound_lower=True,
        bound_upper=True,
    )
    return z_l, z_u


def crown_final_attention_residual_bounds(
    model: TinyMultiBlockViT,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if BoundedModule is None:
        raise RuntimeError(f"auto_LiRPA import failed: {AUTOLIRPA_IMPORT_ERROR!r}")
    h_model = FinalBlockAttentionResidualModule(model).eval()
    bounded = BoundedModule(h_model, (center,), bound_opts={"verbosity": 0, "fixed_reducemax_index": True})
    bounded_x = BoundedTensor(center, PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper))
    h_l, h_u = bounded.compute_bounds(
        x=(bounded_x,),
        method="CROWN",
        bound_lower=True,
        bound_upper=True,
    )
    return h_l, h_u


def direct_classifier_linear_coefficients(
    model: TinyViTBlock,
    label: int,
    target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cls_weight = model.cls.weight
    cls_bias = model.cls.bias
    margin_vec = cls_weight[label] - cls_weight[target]
    margin_bias = cls_bias[label] - cls_bias[target]
    coeff_tokens = margin_vec.unsqueeze(0).expand(model.seq_len, -1) / model.seq_len
    return coeff_tokens, margin_bias


def mlp_preactivation_bounds_from_h_box(
    model: TinyViTBlock,
    h_l: torch.Tensor,
    h_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if h_l.shape[0] != 1:
        raise ValueError("full_mlp vertex_crown currently certifies one example at a time")
    lin1 = model.mlp[0]
    w1 = lin1.weight
    b1 = lin1.bias
    h_l0 = h_l[0]
    h_u0 = h_u[0]
    pre_l = b1 + torch.where(
        w1[None, :, :] >= 0,
        w1[None, :, :] * h_l0[:, None, :],
        w1[None, :, :] * h_u0[:, None, :],
    ).sum(dim=2)
    pre_u = b1 + torch.where(
        w1[None, :, :] >= 0,
        w1[None, :, :] * h_u0[:, None, :],
        w1[None, :, :] * h_l0[:, None, :],
    ).sum(dim=2)
    return pre_l, pre_u


def mlp_preactivation_bounds_vertex(
    model: TinyViTBlock,
    lower: torch.Tensor,
    upper: torch.Tensor,
    score_l: torch.Tensor,
    score_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound each first-layer MLP preactivation directly through attention."""
    if lower.shape[0] != 1:
        raise ValueError("direct full_mlp preactivation bounds currently certify one example at a time")
    lin1 = model.mlp[0]
    w1 = lin1.weight
    b1 = lin1.bias
    mlp_dim = w1.shape[0]
    pre_l = torch.empty(model.seq_len, mlp_dim, device=lower.device, dtype=lower.dtype)
    pre_u = torch.empty_like(pre_l)
    coeff_tokens = torch.zeros(model.seq_len, model.dim, device=lower.device, dtype=lower.dtype)

    for token in range(model.seq_len):
        for hidden in range(mlp_dim):
            coeff_tokens.zero_()
            coeff_tokens[token] = w1[hidden]
            lb = attention_residual_linear_lower(model, lower, upper, coeff_tokens, b1[hidden], score_l, score_u)
            coeff_tokens[token] = -w1[hidden]
            neg_lb = attention_residual_linear_lower(model, lower, upper, coeff_tokens, -b1[hidden], score_l, score_u)
            pre_l[token, hidden] = lb.reshape(()).detach()
            pre_u[token, hidden] = -neg_lb.reshape(()).detach()
    return pre_l, pre_u


def attention_residual_linear_lower_many(
    model: TinyViTBlock,
    lower: torch.Tensor,
    upper: torch.Tensor,
    coeff_tokens: torch.Tensor,
    bias: torch.Tensor,
    score_l: torch.Tensor,
    score_u: torch.Tensor,
) -> torch.Tensor:
    """Batched variant of attention_residual_linear_lower for one input example."""
    if lower.shape[0] != 1:
        raise ValueError("batched attention-residual lower currently supports one example at a time")
    if coeff_tokens.dim() != 3 or coeff_tokens.shape[1:] != (model.seq_len, model.dim):
        raise ValueError(
            f"coeff_tokens must have shape Mx{model.seq_len}x{model.dim}, got {tuple(coeff_tokens.shape)}"
        )

    embed_w = model.embed.weight
    embed_b = model.embed.bias
    pos = model.pos
    lower0 = lower[0]
    upper0 = upper[0]
    objectives = coeff_tokens.shape[0]
    total = torch.as_tensor(bias, device=lower.device, dtype=lower.dtype).reshape(objectives).clone()

    residual_coeff = coeff_tokens.matmul(embed_w)
    residual_bias = ((embed_b.unsqueeze(0) + pos).unsqueeze(0) * coeff_tokens).sum(dim=2)
    residual_lower = residual_bias + torch.where(
        residual_coeff >= 0,
        residual_coeff * lower0.unsqueeze(0),
        residual_coeff * upper0.unsqueeze(0),
    ).sum(dim=2)
    total = total + residual_lower.sum(dim=1)

    if model.out.bias is not None:
        total = total + coeff_tokens.matmul(model.out.bias).sum(dim=1)

    out_direction = coeff_tokens.matmul(model.out.weight)
    for head in range(model.heads):
        start = head * model.head_dim
        end = (head + 1) * model.head_dim
        head_direction = out_direction[:, :, start:end]
        value_w = model.v.weight[start:end, :]
        value_b = model.v.bias[start:end]
        z_direction = head_direction.matmul(value_w)
        token_coeff = z_direction.matmul(embed_w)
        token_bias = (
            torch.einsum("mqd,kd->mqk", z_direction, embed_b.unsqueeze(0) + pos)
            + head_direction.matmul(value_b).unsqueeze(2)
        )
        value_l = token_bias + torch.where(
            token_coeff[:, :, None, :] >= 0,
            token_coeff[:, :, None, :] * lower0[None, None, :, :],
            token_coeff[:, :, None, :] * upper0[None, None, :, :],
        ).sum(dim=3)
        head_score_l = score_l[:, head, :, :].expand(objectives, -1, -1)
        head_score_u = score_u[:, head, :, :].expand(objectives, -1, -1)
        row_l = softmax_box_expectation_min(head_score_l, head_score_u, value_l)
        total = total + row_l.sum(dim=1)

    return total


def mlp_preactivation_bounds_vertex_batched(
    model: TinyViTBlock,
    lower: torch.Tensor,
    upper: torch.Tensor,
    score_l: torch.Tensor,
    score_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound first-layer MLP preactivations with hidden units batched per token."""
    if lower.shape[0] != 1:
        raise ValueError("batched full_mlp preactivation bounds currently certify one example at a time")
    lin1 = model.mlp[0]
    w1 = lin1.weight
    b1 = lin1.bias
    mlp_dim = w1.shape[0]
    pre_l = torch.empty(model.seq_len, mlp_dim, device=lower.device, dtype=lower.dtype)
    pre_u = torch.empty_like(pre_l)
    coeff_tokens = torch.zeros(mlp_dim, model.seq_len, model.dim, device=lower.device, dtype=lower.dtype)

    for token in range(model.seq_len):
        coeff_tokens.zero_()
        coeff_tokens[:, token, :] = w1
        lb = attention_residual_linear_lower_many(model, lower, upper, coeff_tokens, b1, score_l, score_u)
        coeff_tokens[:, token, :] = -w1
        neg_lb = attention_residual_linear_lower_many(model, lower, upper, coeff_tokens, -b1, score_l, score_u)
        pre_l[token] = lb.detach()
        pre_u[token] = -neg_lb.detach()
    return pre_l, pre_u


def mlp_suffix_linear_lower(
    model: TinyViTBlock,
    pre_l: torch.Tensor,
    pre_u: torch.Tensor,
    label: int,
    target: int,
    positive_crossing_slope: str = "auto",
    positive_slope: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CROWN-style affine lower bound on the MLP suffix as a function of h."""
    if model.block_mode != "full_mlp":
        raise ValueError("MLP suffix coefficients require --block-mode full_mlp")
    if positive_crossing_slope not in {"auto", "zero", "identity"}:
        raise ValueError(f"unknown positive crossing ReLU slope policy: {positive_crossing_slope}")
    if positive_slope is not None and positive_slope.shape != pre_l.shape:
        raise ValueError(f"positive_slope must have shape {tuple(pre_l.shape)}, got {tuple(positive_slope.shape)}")

    lin1 = model.mlp[0]
    lin2 = model.mlp[2]
    cls_weight = model.cls.weight
    cls_bias = model.cls.bias
    margin_vec = cls_weight[label] - cls_weight[target]
    margin_bias = cls_bias[label] - cls_bias[target]
    token_margin = margin_vec / model.seq_len

    w1 = lin1.weight
    b1 = lin1.bias
    w2 = lin2.weight
    b2 = lin2.bias

    hidden_coeff = w2.t().matmul(token_margin)
    active = pre_l >= 0
    inactive = pre_u <= 0
    crossing = ~(active | inactive)
    positive_coeff = hidden_coeff >= 0

    slope = torch.zeros_like(pre_l)
    intercept = torch.zeros_like(pre_l)
    slope = torch.where(active, torch.ones_like(slope), slope)

    if positive_slope is not None:
        lower_slope = positive_slope
    elif positive_crossing_slope == "auto":
        lower_slope = (pre_u > -pre_l).to(pre_l.dtype)
    elif positive_crossing_slope == "zero":
        lower_slope = torch.zeros_like(pre_l)
    else:
        lower_slope = torch.ones_like(pre_l)
    pos_crossing = crossing & positive_coeff[None, :]
    slope = torch.where(pos_crossing, lower_slope, slope)

    neg_crossing = crossing & (~positive_coeff[None, :])
    denom = torch.clamp(pre_u - pre_l, min=1e-12)
    upper_slope = pre_u / denom
    upper_intercept = -pre_l * pre_u / denom
    slope = torch.where(neg_crossing, upper_slope, slope)
    intercept = torch.where(neg_crossing, upper_intercept, intercept)

    coeff_tokens = token_margin.unsqueeze(0).expand(model.seq_len, -1).clone()
    coeff_tokens = coeff_tokens + (hidden_coeff[None, :, None] * slope[:, :, None] * w1[None, :, :]).sum(dim=1)
    token_bias = token_margin.matmul(b2) + (hidden_coeff[None, :] * (slope * b1[None, :] + intercept)).sum(dim=1)
    bias = margin_bias + token_bias.sum()
    return coeff_tokens, bias


def attention_residual_linear_lower(
    model: TinyViTBlock,
    lower: torch.Tensor,
    upper: torch.Tensor,
    coeff_tokens: torch.Tensor,
    bias: torch.Tensor,
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    value_l: torch.Tensor | None = None,
    value_u: torch.Tensor | None = None,
) -> torch.Tensor:
    if coeff_tokens.shape != (model.seq_len, model.dim):
        raise ValueError(f"coeff_tokens must have shape {(model.seq_len, model.dim)}, got {tuple(coeff_tokens.shape)}")

    embed_w = model.embed.weight
    embed_b = model.embed.bias
    pos = model.pos
    batch = lower.shape[0]
    value_box_l = value_l
    value_box_u = value_u
    if model.norm_mode == "pre_layernorm":
        if value_box_l is None or value_box_u is None:
            raise ValueError("pre-layernorm Vertex bound requires CROWN bounds on the normalized attention input")
    else:
        value_box_l = value_box_u = None
    total = torch.as_tensor(bias, device=lower.device, dtype=lower.dtype).reshape(1).expand(batch).clone()

    residual_coeff = coeff_tokens.matmul(embed_w)
    residual_bias = ((embed_b.unsqueeze(0) + pos) * coeff_tokens).sum(dim=1)
    total = total + affine_lower(residual_coeff, lower, upper, residual_bias).sum(dim=1)

    if model.out.bias is not None:
        total = total + coeff_tokens.matmul(model.out.bias).sum()

    out_direction = coeff_tokens.matmul(model.out.weight)
    attn_lowers = []
    for head in range(model.heads):
        start = head * model.head_dim
        end = (head + 1) * model.head_dim
        head_direction = out_direction[:, start:end]
        value_w = model.v.weight[start:end, :]
        value_b = model.v.bias[start:end]
        z_direction = head_direction.matmul(value_w)
        if value_box_l is None or value_box_u is None:
            token_coeff = z_direction.matmul(embed_w)
            token_bias = z_direction.matmul((embed_b.unsqueeze(0) + pos).t()) + head_direction.matmul(value_b)[:, None]
            value_coeff_l = token_bias[None, :, :] + torch.where(
                token_coeff[None, :, None, :] >= 0,
                token_coeff[None, :, None, :] * lower[:, None, :, :],
                token_coeff[None, :, None, :] * upper[:, None, :, :],
            ).sum(dim=3)
        else:
            token_bias = head_direction.matmul(value_b)[:, None]
            value_coeff_l = token_bias[None, :, :] + torch.where(
                z_direction[None, :, None, :] >= 0,
                z_direction[None, :, None, :] * value_box_l[:, None, :, :],
                z_direction[None, :, None, :] * value_box_u[:, None, :, :],
            ).sum(dim=3)
        row_l = softmax_box_expectation_min(score_l[:, head, :, :], score_u[:, head, :, :], value_coeff_l)
        attn_lowers.append(row_l.sum(dim=1))

    return total + torch.stack(attn_lowers, dim=0).sum(dim=0)


def attention_residual_linear_lower_zbox(
    block: FinalBlockAdapter,
    z_l: torch.Tensor,
    z_u: torch.Tensor,
    coeff_tokens: torch.Tensor,
    bias: torch.Tensor,
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    value_z_l: torch.Tensor | None = None,
    value_z_u: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lower-bound a linear objective through the final attention-residual block.

    Unlike ``attention_residual_linear_lower``, the input to the attention block
    is already a bounded hidden-state box.  This is the multi-block interface:
    CROWN bounds the prefix into ``z_l,z_u`` and the final score box, while the
    vertex primitive exactly solves each weighted softmax row over that score
    box.
    """
    if coeff_tokens.shape != (block.seq_len, block.dim):
        raise ValueError(f"coeff_tokens must have shape {(block.seq_len, block.dim)}, got {tuple(coeff_tokens.shape)}")

    batch = z_l.shape[0]
    if block.norm_mode == "pre_layernorm":
        if value_z_l is None or value_z_u is None:
            raise ValueError("pre-layernorm final-block Vertex bound requires CROWN bounds on the normalized attention input")
    else:
        value_z_l = z_l
        value_z_u = z_u
    total = torch.as_tensor(bias, device=z_l.device, dtype=z_l.dtype).reshape(1).expand(batch).clone()
    total = total + affine_lower(coeff_tokens, z_l, z_u, torch.zeros(block.seq_len, device=z_l.device, dtype=z_l.dtype)).sum(dim=1)

    if block.out.bias is not None:
        total = total + coeff_tokens.matmul(block.out.bias).sum()

    out_direction = coeff_tokens.matmul(block.out.weight)
    attn_lowers = []
    for head in range(block.heads):
        start = head * block.head_dim
        end = (head + 1) * block.head_dim
        head_direction = out_direction[:, start:end]
        value_w = block.v.weight[start:end, :]
        value_b = block.v.bias[start:end]
        z_direction = head_direction.matmul(value_w)
        token_bias = head_direction.matmul(value_b)[:, None]
        value_l = token_bias[None, :, :] + torch.where(
            z_direction[None, :, None, :] >= 0,
            z_direction[None, :, None, :] * value_z_l[:, None, :, :],
            z_direction[None, :, None, :] * value_z_u[:, None, :, :],
        ).sum(dim=3)
        row_l = softmax_box_expectation_min(score_l[:, head, :, :], score_u[:, head, :, :], value_l)
        attn_lowers.append(row_l.sum(dim=1))

    return total + torch.stack(attn_lowers, dim=0).sum(dim=0)


def optimize_mlp_suffix_vertex_lower(
    model: TinyViTBlock,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    target: int,
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    pre_l: torch.Tensor,
    pre_u: torch.Tensor,
    opt_iters: int,
    opt_lr: float,
) -> torch.Tensor:
    lower = lower.detach()
    upper = upper.detach()
    score_l = score_l.detach()
    score_u = score_u.detach()
    pre_l = pre_l.detach()
    pre_u = pre_u.detach()

    with torch.no_grad():
        candidate_lowers = []
        for slope_policy in ("auto", "zero", "identity"):
            coeff_tokens, bias = mlp_suffix_linear_lower(model, pre_l, pre_u, label, target, slope_policy)
            candidate_lowers.append(attention_residual_linear_lower(model, lower, upper, coeff_tokens, bias, score_l, score_u))
        best_lower = torch.stack(candidate_lowers, dim=0).amax(dim=0).detach()
    if opt_iters <= 0:
        return best_lower

    with torch.no_grad():
        lin2 = model.mlp[2]
        cls_weight = model.cls.weight
        margin_vec = cls_weight[label] - cls_weight[target]
        token_margin = margin_vec / model.seq_len
        hidden_coeff = lin2.weight.t().matmul(token_margin)
        active = pre_l >= 0
        inactive = pre_u <= 0
        positive_crossing = (~(active | inactive)) & (hidden_coeff >= 0)[None, :]
    if not bool(positive_crossing.any()):
        return best_lower

    old_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    slope = torch.nn.Parameter((pre_u > -pre_l).to(pre_l.dtype).clamp(0.0, 1.0))
    optimizer = torch.optim.Adam([slope], lr=opt_lr)
    try:
        for _ in range(opt_iters):
            optimizer.zero_grad(set_to_none=True)
            positive_slope = torch.where(positive_crossing, slope.clamp(0.0, 1.0), torch.zeros_like(slope))
            coeff_tokens, bias = mlp_suffix_linear_lower(
                model,
                pre_l,
                pre_u,
                label,
                target,
                positive_crossing_slope="auto",
                positive_slope=positive_slope,
            )
            lower_bound = attention_residual_linear_lower(model, lower, upper, coeff_tokens, bias, score_l, score_u)
            loss = -lower_bound.sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                slope.clamp_(0.0, 1.0)

        with torch.no_grad():
            positive_slope = torch.where(positive_crossing, slope.clamp(0.0, 1.0), torch.zeros_like(slope))
            coeff_tokens, bias = mlp_suffix_linear_lower(
                model,
                pre_l,
                pre_u,
                label,
                target,
                positive_crossing_slope="auto",
                positive_slope=positive_slope,
            )
            optimized_lower = attention_residual_linear_lower(model, lower, upper, coeff_tokens, bias, score_l, score_u)
            best_lower = torch.maximum(best_lower, optimized_lower.detach())
    finally:
        for parameter, requires_grad in zip(model.parameters(), old_requires_grad):
            parameter.requires_grad_(requires_grad)
    return best_lower


def optimize_mlp_suffix_vertex_lower_zbox(
    block: FinalBlockAdapter,
    z_l: torch.Tensor,
    z_u: torch.Tensor,
    label: int,
    target: int,
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    pre_l: torch.Tensor,
    pre_u: torch.Tensor,
    opt_iters: int,
    opt_lr: float,
) -> torch.Tensor:
    z_l = z_l.detach()
    z_u = z_u.detach()
    score_l = score_l.detach()
    score_u = score_u.detach()
    pre_l = pre_l.detach()
    pre_u = pre_u.detach()

    with torch.no_grad():
        candidate_lowers = []
        for slope_policy in ("auto", "zero", "identity"):
            coeff_tokens, bias = mlp_suffix_linear_lower(block, pre_l, pre_u, label, target, slope_policy)
            candidate_lowers.append(attention_residual_linear_lower_zbox(block, z_l, z_u, coeff_tokens, bias, score_l, score_u))
        best_lower = torch.stack(candidate_lowers, dim=0).amax(dim=0).detach()
    if opt_iters <= 0:
        return best_lower

    with torch.no_grad():
        lin2 = block.mlp[2]
        cls_weight = block.cls.weight
        margin_vec = cls_weight[label] - cls_weight[target]
        token_margin = margin_vec / block.seq_len
        hidden_coeff = lin2.weight.t().matmul(token_margin)
        active = pre_l >= 0
        inactive = pre_u <= 0
        positive_crossing = (~(active | inactive)) & (hidden_coeff >= 0)[None, :]
    if not bool(positive_crossing.any()):
        return best_lower

    params = [*block.q.parameters(), *block.k.parameters(), *block.v.parameters(), *block.out.parameters(), *block.mlp.parameters(), *block.cls.parameters()]
    old_requires_grad = [parameter.requires_grad for parameter in params]
    for parameter in params:
        parameter.requires_grad_(False)
    slope = torch.nn.Parameter((pre_u > -pre_l).to(pre_l.dtype).clamp(0.0, 1.0))
    optimizer = torch.optim.Adam([slope], lr=opt_lr)
    try:
        for _ in range(opt_iters):
            optimizer.zero_grad(set_to_none=True)
            positive_slope = torch.where(positive_crossing, slope.clamp(0.0, 1.0), torch.zeros_like(slope))
            coeff_tokens, bias = mlp_suffix_linear_lower(
                block,
                pre_l,
                pre_u,
                label,
                target,
                positive_crossing_slope="auto",
                positive_slope=positive_slope,
            )
            lower_bound = attention_residual_linear_lower_zbox(block, z_l, z_u, coeff_tokens, bias, score_l, score_u)
            loss = -lower_bound.sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                slope.clamp_(0.0, 1.0)

        with torch.no_grad():
            positive_slope = torch.where(positive_crossing, slope.clamp(0.0, 1.0), torch.zeros_like(slope))
            coeff_tokens, bias = mlp_suffix_linear_lower(
                block,
                pre_l,
                pre_u,
                label,
                target,
                positive_crossing_slope="auto",
                positive_slope=positive_slope,
            )
            optimized_lower = attention_residual_linear_lower_zbox(block, z_l, z_u, coeff_tokens, bias, score_l, score_u)
            best_lower = torch.maximum(best_lower, optimized_lower.detach())
    finally:
        for parameter, requires_grad in zip(params, old_requires_grad):
            parameter.requires_grad_(requires_grad)
    return best_lower


def vertex_margin_lower(
    model: TinyViTBlock,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    target: int,
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    value_l: torch.Tensor | None = None,
    value_u: torch.Tensor | None = None,
    h_l: torch.Tensor | None = None,
    h_u: torch.Tensor | None = None,
    pre_l: torch.Tensor | None = None,
    pre_u: torch.Tensor | None = None,
    full_mlp_suffix_slope_mode: str = "candidates",
    full_mlp_suffix_opt_iters: int = 20,
    full_mlp_suffix_opt_lr: float = 0.1,
) -> torch.Tensor:
    if model.block_mode == "attention_residual":
        coeff_tokens, bias = direct_classifier_linear_coefficients(model, label, target)
        return attention_residual_linear_lower(model, lower, upper, coeff_tokens, bias, score_l, score_u, value_l, value_u)

    elif model.block_mode == "full_mlp":
        if model.norm_mode != "none":
            raise ValueError("pre-layernorm full_mlp Vertex-CROWN is not implemented; use attention_residual or direct CROWN")
        if pre_l is None or pre_u is None:
            if h_l is None or h_u is None:
                raise ValueError("full_mlp vertex_crown requires preactivation bounds or attention-residual bounds")
            pre_l, pre_u = mlp_preactivation_bounds_from_h_box(model, h_l, h_u)
        if full_mlp_suffix_slope_mode == "optimized":
            return optimize_mlp_suffix_vertex_lower(
                model,
                lower,
                upper,
                label,
                target,
                score_l,
                score_u,
                pre_l,
                pre_u,
                full_mlp_suffix_opt_iters,
                full_mlp_suffix_opt_lr,
            )
        if full_mlp_suffix_slope_mode != "candidates":
            raise ValueError(f"unknown full_mlp suffix slope mode: {full_mlp_suffix_slope_mode}")
        candidate_lowers = []
        for slope_policy in ("auto", "zero", "identity"):
            coeff_tokens, bias = mlp_suffix_linear_lower(model, pre_l, pre_u, label, target, slope_policy)
            candidate_lowers.append(attention_residual_linear_lower(model, lower, upper, coeff_tokens, bias, score_l, score_u))
        return torch.stack(candidate_lowers, dim=0).amax(dim=0)

    else:
        raise ValueError(f"unsupported block_mode for vertex_crown: {model.block_mode}")


def final_block_vertex_margin_lower(
    block: FinalBlockAdapter,
    z_l: torch.Tensor,
    z_u: torch.Tensor,
    label: int,
    target: int,
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    value_z_l: torch.Tensor | None = None,
    value_z_u: torch.Tensor | None = None,
    h_l: torch.Tensor | None = None,
    h_u: torch.Tensor | None = None,
    pre_l: torch.Tensor | None = None,
    pre_u: torch.Tensor | None = None,
    full_mlp_suffix_slope_mode: str = "candidates",
    full_mlp_suffix_opt_iters: int = 20,
    full_mlp_suffix_opt_lr: float = 0.1,
) -> torch.Tensor:
    if block.block_mode == "attention_residual":
        coeff_tokens, bias = direct_classifier_linear_coefficients(block, label, target)
        return attention_residual_linear_lower_zbox(
            block, z_l, z_u, coeff_tokens, bias, score_l, score_u, value_z_l, value_z_u
        )

    if block.block_mode == "full_mlp":
        if block.norm_mode != "none":
            raise ValueError("pre-layernorm full_mlp final-block Vertex-CROWN is not implemented; use attention_residual or direct CROWN")
        if pre_l is None or pre_u is None:
            if h_l is None or h_u is None:
                raise ValueError("multi-block full_mlp vertex_crown requires preactivation or final residual bounds")
            pre_l, pre_u = mlp_preactivation_bounds_from_h_box(block, h_l, h_u)
        if full_mlp_suffix_slope_mode == "optimized":
            return optimize_mlp_suffix_vertex_lower_zbox(
                block,
                z_l,
                z_u,
                label,
                target,
                score_l,
                score_u,
                pre_l,
                pre_u,
                full_mlp_suffix_opt_iters,
                full_mlp_suffix_opt_lr,
            )
        if full_mlp_suffix_slope_mode != "candidates":
            raise ValueError(f"unknown full_mlp suffix slope mode: {full_mlp_suffix_slope_mode}")
        candidate_lowers = []
        for slope_policy in ("auto", "zero", "identity"):
            coeff_tokens, bias = mlp_suffix_linear_lower(block, pre_l, pre_u, label, target, slope_policy)
            candidate_lowers.append(attention_residual_linear_lower_zbox(block, z_l, z_u, coeff_tokens, bias, score_l, score_u))
        return torch.stack(candidate_lowers, dim=0).amax(dim=0)

    raise ValueError(f"unsupported block_mode for final-block vertex_crown: {block.block_mode}")


def certify_vertex_target_margins(
    model: TinyViTBlock | TinyMultiBlockViT,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    full_mlp_preactivation_bounds: str = "hbox",
    full_mlp_suffix_slope_mode: str = "candidates",
    full_mlp_suffix_opt_iters: int = 20,
    full_mlp_suffix_opt_lr: float = 0.1,
) -> torch.Tensor:
    with torch.no_grad():
        logits = model(center)
        num_classes = logits.shape[1]
    if isinstance(model, TinyMultiBlockViT) and model.num_blocks > 1:
        if full_mlp_preactivation_bounds != "hbox":
            raise ValueError("multi-block final-block vertex currently supports --full-mlp-preactivation-bounds hbox")
        block = FinalBlockAdapter(model)
        z_l, z_u = crown_final_block_input_bounds(model, center, lower, upper)
        score_l, score_u = crown_final_score_bounds(model, center, lower, upper)
        value_z_l = value_z_u = None
        if model.norm_mode == "pre_layernorm":
            value_z_l, value_z_u = crown_final_attention_input_bounds(model, center, lower, upper)
        h_l = h_u = None
        pre_l = pre_u = None
        if model.block_mode == "full_mlp":
            h_l, h_u = crown_final_attention_residual_bounds(model, center, lower, upper)
        target_lowers = torch.full(
            (num_classes,),
            float("inf"),
            device=lower.device,
            dtype=lower.dtype,
        )
        for target in range(num_classes):
            if target == label:
                continue
            target_lowers[target] = final_block_vertex_margin_lower(
                block,
                z_l,
                z_u,
                label,
                target,
                score_l,
                score_u,
                value_z_l,
                value_z_u,
                h_l,
                h_u,
                pre_l,
                pre_u,
                full_mlp_suffix_slope_mode,
                full_mlp_suffix_opt_iters,
                full_mlp_suffix_opt_lr,
            )
        return target_lowers

    score_l, score_u = crown_score_bounds(model, center, lower, upper)
    value_l = value_u = None
    if model.norm_mode == "pre_layernorm":
        value_l, value_u = crown_attention_input_bounds(model, center, lower, upper)
    h_l = h_u = None
    pre_l = pre_u = None
    if model.block_mode == "full_mlp":
        if full_mlp_preactivation_bounds == "hbox":
            h_l, h_u = crown_attention_residual_bounds(model, center, lower, upper)
        elif full_mlp_preactivation_bounds == "vertex":
            pre_l, pre_u = mlp_preactivation_bounds_vertex(model, lower, upper, score_l, score_u)
        elif full_mlp_preactivation_bounds == "vertex-batched":
            pre_l, pre_u = mlp_preactivation_bounds_vertex_batched(model, lower, upper, score_l, score_u)
        else:
            raise ValueError(f"unknown full_mlp preactivation bound mode: {full_mlp_preactivation_bounds}")
    target_lowers = torch.full(
        (num_classes,),
        float("inf"),
        device=lower.device,
        dtype=lower.dtype,
    )
    for target in range(num_classes):
        if target == label:
            continue
        target_lowers[target] = vertex_margin_lower(
            model,
            lower,
            upper,
            label,
            target,
            score_l,
            score_u,
            value_l,
            value_u,
            h_l,
            h_u,
            pre_l,
            pre_u,
            full_mlp_suffix_slope_mode,
            full_mlp_suffix_opt_iters,
            full_mlp_suffix_opt_lr,
        )
    return target_lowers


def certify_vertex(
    model: TinyViTBlock,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    full_mlp_preactivation_bounds: str = "hbox",
    full_mlp_suffix_slope_mode: str = "candidates",
    full_mlp_suffix_opt_iters: int = 20,
    full_mlp_suffix_opt_lr: float = 0.1,
) -> torch.Tensor:
    target_lowers = certify_vertex_target_margins(
        model,
        center,
        lower,
        upper,
        label,
        full_mlp_preactivation_bounds,
        full_mlp_suffix_slope_mode,
        full_mlp_suffix_opt_iters,
        full_mlp_suffix_opt_lr,
    )
    return target_lowers.min()


def certify_crown_target_margins(
    model: TinyViTBlock,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    method: str,
    alpha_iters: int,
) -> torch.Tensor:
    lbs = crown_margin_lower(model, center, lower, upper, label, method, alpha_iters)
    lbs = lbs.clone()
    lbs[label] = float("inf")
    return lbs


def certify_crown_min_margin(
    model: TinyViTBlock,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    method: str,
    alpha_iters: int,
) -> torch.Tensor:
    return certify_crown_target_margins(model, center, lower, upper, label, method, alpha_iters).min()


def certify_crown_objective_vertex_hybrid(
    model: TinyViTBlock,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    alpha_iters: int,
    full_mlp_preactivation_bounds: str = "hbox",
    full_mlp_suffix_slope_mode: str = "candidates",
    full_mlp_suffix_opt_iters: int = 20,
    full_mlp_suffix_opt_lr: float = 0.1,
) -> torch.Tensor:
    """Sound target-wise terminal hybrid: keep the tighter lower bound per target."""
    _crown_lbs, _objective_vertex_lbs, hybrid_lbs = certify_crown_objective_vertex_target_margins(
        model,
        center,
        lower,
        upper,
        label,
        alpha_iters,
        full_mlp_preactivation_bounds,
        full_mlp_suffix_slope_mode,
        full_mlp_suffix_opt_iters,
        full_mlp_suffix_opt_lr,
    )
    return hybrid_lbs.min()


def certify_crown_objective_vertex_target_margins(
    model: TinyViTBlock,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    alpha_iters: int,
    full_mlp_preactivation_bounds: str = "hbox",
    full_mlp_suffix_slope_mode: str = "candidates",
    full_mlp_suffix_opt_iters: int = 20,
    full_mlp_suffix_opt_lr: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crown_lbs = certify_crown_target_margins(model, center, lower, upper, label, "CROWN", alpha_iters)
    objective_vertex_lbs = certify_vertex_target_margins(
        model,
        center,
        lower,
        upper,
        label,
        full_mlp_preactivation_bounds,
        full_mlp_suffix_slope_mode,
        full_mlp_suffix_opt_iters,
        full_mlp_suffix_opt_lr,
    )
    return crown_lbs, objective_vertex_lbs, torch.maximum(crown_lbs, objective_vertex_lbs)


def certify_target_diagnostics(
    model: TinyViTBlock,
    image: torch.Tensor,
    label: int,
    eps: float,
    patch_size: int,
    alpha_iters: int,
    device: torch.device,
    full_mlp_preactivation_bounds: str,
    full_mlp_suffix_slope_mode: str,
    full_mlp_suffix_opt_iters: int,
    full_mlp_suffix_opt_lr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_l = torch.clamp(image - eps, 0.0, 1.0).unsqueeze(0).to(device)
    img_u = torch.clamp(image + eps, 0.0, 1.0).unsqueeze(0).to(device)
    center = patchify(image.unsqueeze(0), patch_size).to(device)
    lower = patchify(img_l, patch_size)
    upper = patchify(img_u, patch_size)
    return certify_crown_objective_vertex_target_margins(
        model,
        center,
        lower,
        upper,
        label,
        alpha_iters,
        full_mlp_preactivation_bounds,
        full_mlp_suffix_slope_mode,
        full_mlp_suffix_opt_iters,
        full_mlp_suffix_opt_lr,
    )


def certify_one(
    model: TinyViTBlock,
    image: torch.Tensor,
    label: int,
    eps: float,
    patch_size: int,
    method: str,
    alpha_iters: int,
    device: torch.device,
    full_mlp_preactivation_bounds: str,
    full_mlp_suffix_slope_mode: str,
    full_mlp_suffix_opt_iters: int,
    full_mlp_suffix_opt_lr: float,
) -> float:
    img_l = torch.clamp(image - eps, 0.0, 1.0).unsqueeze(0).to(device)
    img_u = torch.clamp(image + eps, 0.0, 1.0).unsqueeze(0).to(device)
    center = patchify(image.unsqueeze(0), patch_size).to(device)
    lower = patchify(img_l, patch_size)
    upper = patchify(img_u, patch_size)

    if method in {"vertex_crown", "objective_vertex_crown"}:
        lb = certify_vertex(
            model,
            center,
            lower,
            upper,
            label,
            full_mlp_preactivation_bounds,
            full_mlp_suffix_slope_mode,
            full_mlp_suffix_opt_iters,
            full_mlp_suffix_opt_lr,
        )
        return float(lb.detach().cpu())

    if method == "crown_objective_vertex_hybrid":
        lb = certify_crown_objective_vertex_hybrid(
            model,
            center,
            lower,
            upper,
            label,
            alpha_iters,
            full_mlp_preactivation_bounds,
            full_mlp_suffix_slope_mode,
            full_mlp_suffix_opt_iters,
            full_mlp_suffix_opt_lr,
        )
        return float(lb.detach().cpu())

    if method in {"CROWN", "alpha-CROWN"}:
        lb = certify_crown_min_margin(model, center, lower, upper, label, method, alpha_iters)
        return float(lb.detach().cpu())

    raise ValueError(f"unknown method {method}")


def run_certification(
    model: TinyViTBlock,
    test_images: torch.Tensor,
    test_tokens: torch.Tensor,
    test_labels: torch.Tensor,
    clean_acc: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[SummaryRow], list[DetailRow], list[TargetDetailRow]]:
    model.eval()
    methods = args.methods
    classes = ",".join(str(c) for c in args.classes)
    summary_rows: list[SummaryRow] = []
    detail_rows: list[DetailRow] = []
    target_detail_rows: list[TargetDetailRow] = []
    printed_error_tracebacks = 0

    with torch.no_grad():
        logits = model(test_tokens[: args.eval_limit].to(device)).detach().cpu()
        preds = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    eval_count = min(args.eval_limit, len(test_labels))
    if args.cert_denominator == "clean":
        correct_indices = torch.nonzero(preds == test_labels[:eval_count], as_tuple=False).flatten()
        if args.cert_limit is not None:
            correct_indices = correct_indices[: args.cert_limit]
        cert_denominator = len(correct_indices)
    else:
        cert_denominator = eval_count if args.cert_limit is None else min(args.cert_limit, eval_count)
        candidate_indices = torch.arange(cert_denominator)
        correct_mask = preds[:cert_denominator] == test_labels[:cert_denominator]
        correct_indices = candidate_indices[correct_mask]
    print(
        f"certifying {len(correct_indices)} correctly classified images "
        f"with denominator={cert_denominator} mode={args.cert_denominator}",
        flush=True,
    )

    pgd_cache: dict[tuple[int, float], float] = {}
    for eps in args.epsilons:
        for idx_tensor in correct_indices:
            idx = int(idx_tensor)
            pgd_cache[(idx, eps)] = pgd_worst_margin(
                model,
                test_images[idx],
                int(test_labels[idx]),
                eps,
                args.patch_size,
                args.pgd_steps,
                args.pgd_restarts,
                args.pgd_step_size,
                device,
            )

        if args.write_target_detail:
            for pos, idx_tensor in enumerate(correct_indices):
                idx = int(idx_tensor)
                label = int(test_labels[idx])
                start = time.time()
                err = ""
                try:
                    crown_lbs, objective_vertex_lbs, hybrid_lbs = certify_target_diagnostics(
                        model,
                        test_images[idx],
                        label,
                        eps,
                        args.patch_size,
                        args.alpha_iters,
                        device,
                        args.full_mlp_preactivation_bounds,
                        args.full_mlp_suffix_slope_mode,
                        args.full_mlp_suffix_opt_iters,
                        args.full_mlp_suffix_opt_lr,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    crown_vals = crown_lbs.detach().cpu()
                    objective_vertex_vals = objective_vertex_lbs.detach().cpu()
                    hybrid_vals = hybrid_lbs.detach().cpu()
                    crown_worst = int(crown_vals.argmin().item())
                    objective_vertex_worst = int(objective_vertex_vals.argmin().item())
                    hybrid_worst = int(hybrid_vals.argmin().item())
                except Exception as exc:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    err = repr(exc)
                    if printed_error_tracebacks < args.max_error_tracebacks:
                        print(
                            f"target_detail_error eps={eps} index={idx} label={label} error={err}",
                            flush=True,
                        )
                        traceback.print_exc()
                        printed_error_tracebacks += 1
                    crown_vals = torch.full((num_classes,), float("nan"))
                    objective_vertex_vals = torch.full((num_classes,), float("nan"))
                    hybrid_vals = torch.full((num_classes,), float("nan"))
                    crown_worst = objective_vertex_worst = hybrid_worst = -1
                elapsed = time.time() - start

                for target in range(num_classes):
                    if target == label:
                        continue
                    crown_lower = float(crown_vals[target])
                    objective_vertex_lower = float(objective_vertex_vals[target])
                    hybrid_lower = float(hybrid_vals[target])
                    if not (math.isfinite(crown_lower) and math.isfinite(objective_vertex_lower)):
                        winner = "error"
                    elif objective_vertex_lower > crown_lower + 1e-7:
                        winner = "vertex"
                    elif crown_lower > objective_vertex_lower + 1e-7:
                        winner = "crown"
                    else:
                        winner = "tie"
                    target_detail_rows.append(
                        TargetDetailRow(
                            dataset=args.dataset,
                            classes=classes,
                            seed=args.seed,
                            block_mode=args.block_mode,
                            norm_mode=args.norm_mode,
                            layernorm_eps=args.layernorm_eps,
                            num_blocks=args.num_blocks,
                            patch_size=args.patch_size,
                            dim=args.dim,
                            heads=args.heads,
                            mlp_dim=args.mlp_dim,
                            epsilon=eps,
                            index=idx,
                            label=label,
                            pred=int(preds[idx]),
                            target=target,
                            pgd_margin=pgd_cache[(idx, eps)],
                            crown_lower=crown_lower,
                            objective_vertex_lower=objective_vertex_lower,
                            hybrid_lower=hybrid_lower,
                            winner=winner,
                            crown_worst_target=int(target == crown_worst),
                            objective_vertex_worst_target=int(target == objective_vertex_worst),
                            hybrid_worst_target=int(target == hybrid_worst),
                            crown_certified_target=int(crown_lower > 0.0),
                            objective_vertex_certified_target=int(objective_vertex_lower > 0.0),
                            hybrid_certified_target=int(hybrid_lower > 0.0),
                            elapsed_sec=elapsed,
                            error=err,
                        )
                    )
                if (pos + 1) % max(1, len(correct_indices) // 5) == 0:
                    print(
                        f"eps={eps} target_detail done={pos + 1}/{len(correct_indices)}",
                        flush=True,
                    )

        for method in methods:
            lowers = []
            elapsed_total = 0.0
            errors = 0
            for pos, idx_tensor in enumerate(correct_indices):
                idx = int(idx_tensor)
                label = int(test_labels[idx])
                start = time.time()
                err = ""
                try:
                    lb = certify_one(
                        model,
                        test_images[idx],
                        label,
                        eps,
                        args.patch_size,
                        method,
                        args.alpha_iters,
                        device,
                        args.full_mlp_preactivation_bounds,
                        args.full_mlp_suffix_slope_mode,
                        args.full_mlp_suffix_opt_iters,
                        args.full_mlp_suffix_opt_lr,
                    )
                except Exception as exc:
                    lb = float("nan")
                    err = repr(exc)
                    errors += 1
                    if printed_error_tracebacks < args.max_error_tracebacks:
                        print(
                            f"certify_error eps={eps} method={method} index={idx} label={label} error={err}",
                            flush=True,
                        )
                        traceback.print_exc()
                        printed_error_tracebacks += 1
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                elapsed_total += elapsed
                lowers.append(lb)
                detail_rows.append(
                    DetailRow(
                        dataset=args.dataset,
                        classes=classes,
                        seed=args.seed,
                        block_mode=args.block_mode,
                        norm_mode=args.norm_mode,
                        layernorm_eps=args.layernorm_eps,
                        num_blocks=args.num_blocks,
                        patch_size=args.patch_size,
                        dim=args.dim,
                        heads=args.heads,
                        mlp_dim=args.mlp_dim,
                        epsilon=eps,
                        method=method,
                        index=idx,
                        label=label,
                        pred=int(preds[idx]),
                        pgd_margin=pgd_cache[(idx, eps)],
                        lower_bound=lb,
                        certified=int(lb > 0.0),
                        elapsed_sec=elapsed,
                        error=err,
                    )
                )
                if (pos + 1) % max(1, len(correct_indices) // 5) == 0:
                    print(
                        f"eps={eps} method={method} done={pos + 1}/{len(correct_indices)}",
                        flush=True,
                    )
            finite = torch.tensor([v for v in lowers if math.isfinite(v)])
            images = cert_denominator
            cert_acc = float((finite > 0).float().sum().item() / max(1, cert_denominator))
            pgd_acc = float(
                sum(pgd_cache[(int(i), eps)] > 0.0 for i in correct_indices) / max(1, cert_denominator)
            )
            summary = SummaryRow(
                dataset=args.dataset,
                classes=classes,
                seed=args.seed,
                block_mode=args.block_mode,
                norm_mode=args.norm_mode,
                layernorm_eps=args.layernorm_eps,
                num_blocks=args.num_blocks,
                patch_size=args.patch_size,
                dim=args.dim,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                epsilon=eps,
                method=method,
                clean_acc=clean_acc,
                pgd_acc=pgd_acc,
                cert_acc=cert_acc,
                mean_lower=float(finite.mean().item()) if finite.numel() else float("nan"),
                median_lower=float(finite.median().item()) if finite.numel() else float("nan"),
                images=images,
                elapsed_sec=elapsed_total,
                sec_per_image=elapsed_total / max(1, len(lowers)),
                errors=errors,
            )
            summary_rows.append(summary)
            print(summary, flush=True)
    return summary_rows, detail_rows, target_detail_rows


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=dataset_choices(), default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--eval-limit", type=int, default=2000)
    parser.add_argument("--cert-limit", type=int, default=100)
    parser.add_argument(
        "--cert-denominator",
        choices=["clean", "eval"],
        default="clean",
        help=(
            "clean certifies the first cert-limit clean-correct examples; "
            "eval uses the first cert-limit eval examples as the denominator and "
            "counts misclassified examples as uncertified."
        ),
    )
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--block-mode", choices=["attention_residual", "full_mlp"], default="attention_residual")
    parser.add_argument("--norm-mode", choices=["none", "pre_layernorm"], default="none")
    parser.add_argument("--layernorm-eps", type=float, default=1e-5)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--mlp-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    parser.add_argument("--methods", nargs="+", default=["CROWN", "vertex_crown"])
    parser.add_argument("--full-mlp-preactivation-bounds", choices=["hbox", "vertex", "vertex-batched"], default="hbox")
    parser.add_argument("--full-mlp-suffix-slope-mode", choices=["candidates", "optimized"], default="candidates")
    parser.add_argument("--full-mlp-suffix-opt-iters", type=int, default=20)
    parser.add_argument("--full-mlp-suffix-opt-lr", type=float, default=0.1)
    parser.add_argument("--write-target-detail", action="store_true")
    parser.add_argument("--alpha-iters", type=int, default=20)
    parser.add_argument("--max-error-tracebacks", type=int, default=3)
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--pgd-restarts", type=int, default=2)
    parser.add_argument("--pgd-step-size", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(
        f"device={device} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"dataset={args.dataset} classes={args.classes} block_mode={args.block_mode} "
        f"norm_mode={args.norm_mode} num_blocks={args.num_blocks} heads={args.heads}",
        flush=True,
    )
    if BoundedModule is None:
        print(f"auto_LiRPA import failed: {AUTOLIRPA_IMPORT_ERROR!r}", flush=True)

    train_images, train_labels, test_images, test_labels = load_dataset(args.data_dir, args.dataset)
    train_images, train_labels = select_classes(train_images, train_labels, args.classes, args.train_limit)
    test_images, test_labels = select_classes(test_images, test_labels, args.classes, args.eval_limit)
    train_tokens = patchify(train_images, args.patch_size)
    test_tokens = patchify(test_images, args.patch_size)

    if args.num_blocks == 1:
        model = TinyViTBlock(
            patch_dim=train_tokens.shape[-1],
            dim=args.dim,
            heads=args.heads,
            num_classes=len(args.classes),
            seq_len=train_tokens.shape[1],
            block_mode=args.block_mode,
            norm_mode=args.norm_mode,
            layernorm_eps=args.layernorm_eps,
            mlp_dim=args.mlp_dim,
        )
    else:
        model = TinyMultiBlockViT(
            patch_dim=train_tokens.shape[-1],
            dim=args.dim,
            heads=args.heads,
            num_classes=len(args.classes),
            seq_len=train_tokens.shape[1],
            block_mode=args.block_mode,
            norm_mode=args.norm_mode,
            layernorm_eps=args.layernorm_eps,
            mlp_dim=args.mlp_dim,
            num_blocks=args.num_blocks,
        )
    clean_acc = train_model(model, train_tokens, train_labels, test_tokens, test_labels, args, device)
    print(f"final_clean_acc={clean_acc:.4f}", flush=True)

    summary_rows, detail_rows, target_detail_rows = run_certification(
        model, test_images, test_tokens, test_labels, clean_acc, args, device
    )
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary_rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_detail.csv"), detail_rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_target_detail.csv"), target_detail_rows)


if __name__ == "__main__":
    main()
