import numpy as np

from .distributions import DegreeDistribution, JDistribution
from .config import PopulationConfig
from .kernels import onsite_phi, _update_one_nb

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


class CavityPopulation:
    """Minimal state; updates run via kernels in simulation/replica modules."""
    def __init__(
        self,
        degree: DegreeDistribution,
        couplings: JDistribution,
        cfg: PopulationConfig,
        population: np.ndarray | None = None,
    ):
        self.degree = degree
        self.couplings = couplings
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        G = cfg.G
        self.grid = np.linspace(0.0, 1.0, G, dtype=cfg.dtype)
        self.n = self.grid

        M = cfg.M
        if population is None:
            a = np.array(0.5, dtype=cfg.dtype)
            b = (-1.0 + 0.2 * self.rng.normal(size=M)).astype(cfg.dtype, copy=False)
            c = (0.1 * self.rng.normal(size=M)).astype(cfg.dtype, copy=False)
            self.population = (a * self.n[None] ** 2 + b[:, None] * self.n[None] + c[:, None]).astype(cfg.dtype, copy=False)
            if cfg.recenter:
                self.population -= self.population.min(axis=1, keepdims=True).astype(cfg.dtype, copy=False)
        else:
            arr = np.asarray(population, dtype=cfg.dtype)
            if arr.ndim != 2 or arr.shape[1] != G:
                raise ValueError(f"population must have shape (pool_size, {G})")
            self.population = np.array(arr, dtype=cfg.dtype, copy=True)
            if cfg.recenter:
                self.population -= self.population.min(axis=1, keepdims=True).astype(cfg.dtype, copy=False)

    def abundances(
        self,
        n_samples: int,
        *,
        seed: int | None = None,
        progress_every: int = 1000,
        c_mean: float | None = None,
    ) -> np.ndarray:
        """
        Sample abundance marginals from the current field pool.

        Parameters
        ----------
        n_samples:
            Number of marginals to sample.
        seed:
            RNG seed. Defaults to cfg.seed + 1.
        progress_every:
            Progress-bar update cadence (samples per chunk).
        c_mean:
            Mean of Poisson degree draws. Defaults to the configured degree mean.
        """
        n_samples = int(n_samples)
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        progress_every = max(1, int(progress_every))

        rng = np.random.default_rng(int(self.cfg.seed) + 1 if seed is None else int(seed))
        dtype = self.population.dtype

        if c_mean is None:
            c_mean = float(self.degree.mean(int(self.cfg.k_max)))
        c_mean = max(0.0, float(c_mean))

        x = self.n.astype(dtype, copy=False)
        y = self.grid.astype(dtype, copy=False)
        phi = onsite_phi(x).astype(dtype, copy=False)
        scratch = np.empty_like(phi)
        hbuf = np.empty_like(phi)
        pool = self.population
        pool_size = int(pool.shape[0])

        samples = np.empty(n_samples, dtype=np.float64)
        pbar = tqdm(
            total=n_samples,
            desc="Cavity sample marginals",
            unit="samp",
            leave=True,
            disable=not bool(self.cfg.verbose),
        )
        done = 0
        while done < n_samples:
            chunk = min(progress_every, n_samples - done)
            for s in range(chunk):
                d = int(rng.poisson(c_mean))
                if d > 0:
                    idx = rng.integers(0, pool_size, size=d, dtype=np.int64)
                    Js = self.couplings.sample(rng, size=d).astype(dtype, copy=False)
                else:
                    idx = np.empty(0, dtype=np.int64)
                    Js = np.empty(0, dtype=dtype)

                _update_one_nb(pool, phi, x, y, idx, Js, scratch, hbuf, bool(self.cfg.recenter))
                samples[done + s] = float(x[int(np.argmin(hbuf))])
            done += chunk
            pbar.update(chunk)
        pbar.close()
        return samples
