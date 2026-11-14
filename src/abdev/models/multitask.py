"""Shared-trunk multi-task MLP.

Predicts all assay targets jointly from the fused feature vector. Joint training lets
correlated assays regularize one another -- important when each task has only ~250 labels.
Missing labels are masked per-sample in the loss. PyTorch is optional; if unavailable, the
ensemble simply omits this head.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class MultiTaskMLP:
    def __init__(self, task_names: List[str], hidden=(512, 128), dropout: float = 0.3,
                 lr: float = 1e-3, weight_decay: float = 1e-3, epochs: int = 300,
                 batch_size: int = 64, random_state: int = 42):
        self.task_names = task_names
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _build(self, in_dim: int, out_dim: int):
        import torch.nn as nn
        layers, d = [], in_dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X: (n, d). Y: (n, t) with NaN for missing labels (masked in loss)."""
        import torch
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-6
        self.y_mean_ = np.nanmean(Y, axis=0)
        self.y_std_ = np.nanstd(Y, axis=0) + 1e-6
        Xn = (X - self.x_mean_) / self.x_std_
        Yn = (Y - self.y_mean_) / self.y_std_
        mask = ~np.isnan(Yn)
        Yn = np.nan_to_num(Yn)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net_ = self._build(X.shape[1], Y.shape[1]).to(device)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        xt = torch.tensor(Xn, device=device)
        yt = torch.tensor(Yn, device=device)
        mt = torch.tensor(mask.astype(np.float32), device=device)

        n = X.shape[0]
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.batch_size):
                idx = perm[i: i + self.batch_size]
                if idx.numel() < 2:
                    continue
                opt.zero_grad()
                pred = self.net_(xt[idx])
                diff = (pred - yt[idx]) ** 2 * mt[idx]
                loss = diff.sum() / mt[idx].sum().clamp(min=1.0)
                loss.backward()
                opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Xn = (np.asarray(X, dtype=np.float32) - self.x_mean_) / self.x_std_
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.tensor(Xn, device=device)).cpu().numpy()
        return out * self.y_std_ + self.y_mean_  # (n, t) in original units

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False
