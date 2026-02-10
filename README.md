# cavity

Cavity population dynamics simulations.

## Quick start

```bash
python -c "from cavity import DegreeDistribution, JDistribution, PopulationConfig, cavity_simulation; print(cavity_simulation(DegreeDistribution.poisson(3.0), JDistribution(), PopulationConfig(T=10, verbose=False)).shape)"
```

## Modules

- `cavity.distributions`: degree and coupling distributions
- `cavity.config`: configuration dataclass
- `cavity.population`: population state container
- `cavity.kernels`: numba-accelerated kernels and helpers
- `cavity.simulation`: single-population simulation and warm-up pool
- `cavity.replica`: replica traces
