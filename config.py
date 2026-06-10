from __future__ import annotations

from dataclasses import dataclass

from core import SSSBParams


@dataclass(frozen=True)
class SimConfig:
    """
    Full-field simulation runs (simulate_sssb) that generate snapshot figures per case.
    """
    params: SSSBParams = SSSBParams(
        grid_N=(100, 200, 50),
        p=0.01,
        q=1,
        gamma_J=0.1,
        k_J=0.5,
        D=0.05,
        S0=0,
        Ncap=10,
        dt=0.02,
        n_steps=800,
        seed=0,
        verbose=True,
        verbose_freq=250,
    )
    n_snaps: int = 5


@dataclass(frozen=True)
class AdoptConfig:
    """
    Adoption-curve runs (simulate_sssb_adoption_curve) that only return (t, A).
    Usually can use larger grids than the full-field runs because memory is tiny.
    """
    params: SSSBParams = SSSBParams(
        grid_N=(10, 6, 6),
        p=0.01,
        q=1,
        gamma_J=0.1,
        k_J=0.5,
        D=0.05,
        S0=0.0,
        Ncap=5,
        dt=0.1,
        n_steps=300,
        seed=0,
        verbose=True,
        verbose_freq=250,
    )
    n_runs: int = 100


@dataclass(frozen=True)
class MeanConfig:
    """
    Median runs (simulate_sssb_selected_snapshots): long-running Monte Carlo.
    Keep grid_N[0] moderate or this will be expensive.
    """
    params: SSSBParams = SSSBParams(
        grid_N=(120, 30, 50), #120
        p=0.05,
        q=5,
        gamma_J=0.1,
        k_J=0.01,
        D=1,
        S0=0,
        Ncap=20,
        dt=0.02,
        n_steps=200,
        seed=0,
        verbose=False,
        verbose_freq=10000,
    )
    n_runs: int = 2000
    n_snaps: int = 10
    dim: int = 2
    periodic: bool = False


SIM = SimConfig()
ADOPT = AdoptConfig()
MEAN = MeanConfig()