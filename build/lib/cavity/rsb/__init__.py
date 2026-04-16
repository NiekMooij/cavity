"""
cavity.rsb — One-step Replica Symmetry Breaking (1RSB) extensions.

Public API
----------
Population1RSB
    Pool of 1RSB cavity fields; the 1RSB counterpart of CavityPopulation.

estimate_phi(population, beta, *, n_mc, seed) -> dict
    Estimate the Bethe free entropy Φ(β) via Monte Carlo.

estimate_site_observables(population, *, n_mc, seed) -> dict
    Estimate diversity and mean abundance from the 1RSB pool.

compute_complexity(beta_values, phi_values) -> dict
    Legendre transform Φ(β) → complexity curve Σ(ε).

rs_energy(c, mu, g) -> float
    Analytic replica-symmetric energy density for a c-regular graph (diagnostic).
"""
from __future__ import annotations

from .population import Population1RSB
from .free_energy import estimate_phi, estimate_site_observables
from .complexity import compute_complexity, rs_energy

__all__ = [
    "Population1RSB",
    "estimate_phi",
    "estimate_site_observables",
    "compute_complexity",
    "rs_energy",
]
