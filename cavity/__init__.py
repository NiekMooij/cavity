"""Cavity package public API."""

# RS (replica-symmetric) core
from .distributions import DegreeDistribution, JDistribution
from .config import PopulationConfig
from .population import CavityPopulation
from .simulation import cavity_simulation, warm_up_pool, lv_integration
from .replica import replica_h_traces

# 1RSB extensions (cavity.rsb subpackage)
RSB_AVAILABLE = False
RSB_IMPORT_ERROR = None

try:
    from .rsb import Population1RSB, estimate_phi, estimate_site_observables, compute_complexity, rs_energy
    RSB_AVAILABLE = True
except Exception as exc:
    RSB_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

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
    "RSB_AVAILABLE",
    "RSB_IMPORT_ERROR",
]

if RSB_AVAILABLE:
    __all__.extend([
        "Population1RSB",
        "estimate_phi",
        "estimate_site_observables",
        "compute_complexity",
        "rs_energy",
    ])
