"""Cavity package public API."""

# RS (replica-symmetric) core
from .distributions import DegreeDistribution, JDistribution
from .config import PopulationConfig
from .population import CavityPopulation
from .simulation import cavity_simulation, warm_up_pool, lv_integration
from .replica import replica_h_traces

# 1RSB extensions (cavity.rsb subpackage)
from .rsb import Population1RSB, estimate_phi, estimate_site_observables, compute_complexity, rs_energy

__all__ = [
    # RS core
    "DegreeDistribution",
    "JDistribution",
    "PopulationConfig",
    "CavityPopulation",
    "cavity_simulation",
    "warm_up_pool",
    "lv_integration",
    "replica_h_traces",
    # 1RSB
    "Population1RSB",
    "estimate_phi",
    "estimate_site_observables",
    "compute_complexity",
    "rs_energy",
]
