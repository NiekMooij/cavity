from __future__ import annotations

import numpy as np


def _norm_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("bad pmf")
    s = p.sum()
    if s <= 0:
        raise ValueError("bad sum")
    return p / s


class DegreeDistribution:
    """
    Degree distribution descriptor.

    kind:
      - "poisson"         : Poisson with mean c
      - "random_regular"  : k-regular with k=int(c)
      - "custom"          : user-provided pmf (and optional c)
    """
    def __init__(self, kind: str = "poisson", c: float = 3.0, pmf: np.ndarray | None = None):
        self.kind = str(kind)
        self.c = float(c)
        self.pmf = None if pmf is None else np.asarray(pmf, float)

    @classmethod
    def poisson(cls, mean_c: float) -> "DegreeDistribution":
        return cls(kind="poisson", c=float(mean_c))

    @classmethod
    def random_regular(cls, k: int) -> "DegreeDistribution":
        """k-regular graph (fixed degree k)."""
        return cls(kind="random_regular", c=float(k))

    @classmethod
    def custom(cls, pmf: np.ndarray, c: float | None = None) -> "DegreeDistribution":
        pmf = np.asarray(pmf, float)
        c_val = float(c) if c is not None else float(np.sum(np.arange(len(pmf), dtype=float) * _norm_p(pmf)))
        return cls(kind="custom", pmf=pmf, c=c_val)

    def pmf_array(self, k_max: int = 50) -> np.ndarray:
        if self.kind == "poisson":
            k = np.arange(k_max + 1, dtype=int)
            p = np.empty(k.size, float)
            p[0] = np.exp(-self.c)
            for i in range(1, k.size):
                p[i] = p[i - 1] * self.c / i
            return _norm_p(p)
        if self.kind == "random_regular":
            k_max = max(int(k_max), int(self.c))
            p = np.zeros(k_max + 1, float)
            p[int(self.c)] = 1.0
            return p
        if self.kind == "custom":
            if self.pmf is None:
                raise ValueError("need pmf")
            return _norm_p(self.pmf)
        raise ValueError(f"unknown degree distribution kind: {self.kind}")

    def mean(self, k_max: int = 50) -> float:
        p = self.pmf_array(k_max)
        k = np.arange(p.size, dtype=float)
        return float((k * p).sum())

    def sample_k(self, rng: np.random.Generator, size_bias: bool, k_max: int = 50) -> int:
        p = self.pmf_array(k_max)
        k = np.arange(p.size, dtype=int)
        if size_bias:
            c = self.mean(k_max)
            if c <= 0:
                return 0
            q = _norm_p(p * k / c)
            return int(rng.choice(k, p=q))
        return int(rng.choice(k, p=p))


class JDistribution:
    """ Return NEGATIVE couplings by convention ( -X with X ≥ 0 distributions ). """
    def __init__(
        self,
        kind: str = "abs_gaussian",
        *,
        normal_mu: float = 0.0,
        normal_sigma: float = 0.1,
        uniform_low: float = 0.0,
        uniform_high: float = 1.0,
        gamma_shape: float = 2.0,
        gamma_scale: float = 1.0,
    ):
        if kind not in {"uniform_pos", "abs_gaussian", "trunc_gaussian_0", "gamma_pos"}:
            raise ValueError("kind must be one of {'uniform_pos','abs_gaussian','trunc_gaussian_0','gamma_pos'}")
        self.kind = kind
        self.normal_mu = None if normal_mu is None else float(normal_mu)
        self.normal_sigma = None if normal_sigma is None else float(normal_sigma)
        self.uniform_low = None if uniform_low is None else float(uniform_low)
        self.uniform_high = None if uniform_high is None else float(uniform_high)
        self.gamma_shape = None if gamma_shape is None else float(gamma_shape)
        self.gamma_scale = None if gamma_scale is None else float(gamma_scale)
        if self.kind in {"abs_gaussian", "trunc_gaussian_0"} and not (self.normal_sigma >= 0):
            raise ValueError("normal_sigma must be non-negative")
        if self.kind == "uniform_pos" and not (0.0 <= self.uniform_low < self.uniform_high):
            raise ValueError("Require 0 <= low < high for 'uniform_pos'")
        if self.kind == "gamma_pos" and not (self.gamma_shape > 0 and self.gamma_scale > 0):
            raise ValueError("Require shape>0, scale>0 for 'gamma_pos'")

    def sample(self, rng: np.random.Generator, size=None) -> np.ndarray:
        if self.kind == "uniform_pos":
            return -rng.uniform(self.uniform_low, self.uniform_high, size=size)
        if self.kind == "abs_gaussian":
            return -np.abs(rng.normal(self.normal_mu, self.normal_sigma, size=size))
        if self.kind == "gamma_pos":
            return -rng.gamma(shape=self.gamma_shape, scale=self.gamma_scale, size=size)
        if self.kind == "trunc_gaussian_0":
            if size is None:
                x = rng.normal(self.normal_mu, self.normal_sigma)
                while x < 0.0:
                    x = rng.normal(self.normal_mu, self.normal_sigma)
                return -np.array(x, dtype=float)
            x = rng.normal(self.normal_mu, self.normal_sigma, size=size)
            mask = (x < 0.0)
            while np.any(mask):
                x[mask] = rng.normal(self.normal_mu, self.normal_sigma, size=int(mask.sum()))
                mask = (x < 0.0)
            return -x
        raise RuntimeError("Unhandled JDistribution kind")
