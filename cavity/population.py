import numpy as np

from .distributions import DegreeDistribution, JDistribution
from .config import PopulationConfig


class CavityPopulation:
    """Minimal state; updates run via kernels in simulation/replica modules."""
    def __init__(self, degree: DegreeDistribution, couplings: JDistribution, cfg: PopulationConfig):
        self.degree = degree
        self.couplings = couplings
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        G = cfg.G
        self.grid = np.linspace(0.0, 1.0, G, dtype=cfg.dtype)
        self.n = self.grid

        M = cfg.M
        a = np.array(0.5, dtype=cfg.dtype)
        b = (-1.0 + 0.2 * self.rng.normal(size=M)).astype(cfg.dtype, copy=False)
        c = (0.1 * self.rng.normal(size=M)).astype(cfg.dtype, copy=False)
        self.population = (a * self.n[None] ** 2 + b[:, None] * self.n[None] + c[:, None]).astype(cfg.dtype, copy=False)
        if cfg.recenter:
            self.population -= self.population.min(axis=1, keepdims=True).astype(cfg.dtype, copy=False)
