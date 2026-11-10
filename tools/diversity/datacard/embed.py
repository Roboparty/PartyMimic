"""Frozen window embedding: z-score (clipped at +-8 sd) + PCA(32), fitted once on a declared reference pool.

Distances between datasets are only comparable under the same embedding, so an embedding is fitted on a pool that
covers everything to be compared (e.g. all training versions + the official LAFAN set) and then frozen. Feature
dimensions depend on the joint count, so embeddings are per robot; ``check_compatible`` enforces robot, feature
version and dimension.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .features import FEATURE_VERSION, FeatureSet


@dataclass
class Embedding:
    mu: np.ndarray
    sd: np.ndarray
    components: np.ndarray
    pca_mean: np.ndarray
    explained_variance_ratio: np.ndarray
    clip: float = 8.0
    meta: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for a in (self.mu, self.sd, self.components, self.pca_mean):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]

    def zscore(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mu) / self.sd, -self.clip, self.clip).astype(np.float32)

    def transform(self, X: np.ndarray, batch: int = 262144) -> np.ndarray:
        """(n, F) window features -> (n, dim) float32, in the same arithmetic as ``sklearn.decomposition.PCA``."""
        out = np.empty((len(X), self.dim), np.float32)
        shift = self.pca_mean.reshape(1, -1) @ self.components.T
        for s in range(0, len(X), batch):
            y = self.zscore(X[s:s + batch]) @ self.components.T
            y -= shift
            out[s:s + batch] = y
        return out

    def check_compatible(self, fs: FeatureSet) -> None:
        m = self.meta
        if fs.robot != m["robot"] or fs.meta["feature_version"] != m["feature_version"] or fs.X.shape[1] != len(self.mu):
            raise ValueError(f"feature set {fs.name} ({fs.robot}, {fs.meta['feature_version']}, dim {fs.X.shape[1]}) does "
                             f"not match embedding ({m['robot']}, {m['feature_version']}, dim {len(self.mu)})")
        if (fs.meta["window_s"], fs.meta["stride_s"]) != (m["window_s"], m["stride_s"]):
            raise ValueError(f"feature set {fs.name}: window settings differ from the embedding")

    def embed(self, fs: FeatureSet) -> np.ndarray:
        self.check_compatible(fs)
        return self.transform(fs.X)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mu=self.mu, sd=self.sd, components=self.components, pca_mean=self.pca_mean,
                 explained_variance_ratio=self.explained_variance_ratio, clip=np.float64(self.clip),
                 meta_json=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "Embedding":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["mu"], z["sd"], z["components"], z["pca_mean"], z["explained_variance_ratio"],
                       float(z["clip"]), json.loads(str(z["meta_json"])))


def pool_summary(fs: FeatureSet) -> dict:
    groups = {}
    for g in np.unique(fs.clip_group):
        m = fs.clip_group == g
        groups[str(g)] = {"clips": int(m.sum()), "hours": float(fs.clip_hours()[m].sum()),
                          "windows": int(np.isin(fs.clip, np.flatnonzero(m)).sum())}
    return {"name": fs.name, "clips": fs.n_clips, "windows": fs.n_windows, "hours": fs.hours(), "groups": groups}


def fit_embedding(pool: FeatureSet, n_components: int = 32, clip: float = 8.0, seed: int = 0,
                  description: str = "") -> Embedding:
    """Fit on the reference pool windows; ``description`` should say what the pool is (it is stored with the model)."""
    t0 = time.time()
    X = pool.X
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    z = np.clip((X - mu) / sd, -clip, clip).astype(np.float32)
    pca = PCA(n_components, svd_solver="randomized", random_state=seed).fit(z)
    meta = {"feature_version": FEATURE_VERSION, "robot": pool.robot, "feature_dim": int(X.shape[1]),
            "window_s": pool.meta["window_s"], "stride_s": pool.meta["stride_s"], "n_components": n_components,
            "clip": clip, "seed": seed, "pool_description": description, "pool": pool_summary(pool),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "fit_seconds": round(time.time() - t0, 1)}
    return Embedding(mu, sd, pca.components_, pca.mean_, pca.explained_variance_ratio_, clip, meta)
