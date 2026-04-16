import numpy as np


class PopulationConfig:
    def __init__(
        self,
        G: int = 401,
        M: int = 1000,
        T: int = 20000,
        k_max: int = 50,
        seed: int = 42,
        verbose: bool = True,
        recenter: bool = True,
        dtype=np.float32,
    ):
        self.G = int(G)
        self.M = int(M)
        self.T = int(T)
        self.k_max = int(k_max)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.recenter = bool(recenter)
        self.dtype = dtype
