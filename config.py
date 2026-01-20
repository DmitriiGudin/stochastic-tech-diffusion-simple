from __future__ import annotations

from dataclasses import dataclass

from core import SSSBParams


@dataclass(frozen=True)
class SimConfig:
    """
    Full-field simulation runs (simulate_sssb) that generate snapshot figures per case.
    """
    params: SSSBParams = SSSBParams(
        grid_N=(100, 100, 50),
        p=0.01,
        q=1.0,
        gamma_J=0.1,
        k_J=0.5,
        D=0.05,
        S0=0.0,
        Ncap=10,
        dt=0.02,
        n_steps=1000,
        seed=0,
        verbose=True,
        verbose_freq=250,
    )
    n_snaps: int = 3


@dataclass(frozen=True)
class AdoptConfig:
    """
    Adoption-curve runs (simulate_sssb_adoption_curve) that only return (t, A).
    Usually can use larger grids than the full-field runs because memory is tiny.
    """
    params: SSSBParams = SSSBParams(
        grid_N=(50, 30, 20),
        p=0.01,
        q=1.0,
        gamma_J=0.1,
        k_J=0.5,
        D=0.05,
        S0=0.0,
        Ncap=10,
        dt=0.02,
        n_steps=1200,
        seed=0,
        verbose=True,
        verbose_freq=250,
    )


@dataclass(frozen=True)
class MeanConfig:
    """
    Median runs (simulate_sssb_selected_snapshots): long-running Monte Carlo.
    Keep grid_N[0] moderate or this will be expensive.
    """
    params: SSSBParams = SSSBParams(
        grid_N=(50, 100, 50), #120
        p=0.02,
        q=0.3,
        gamma_J=0.5,
        k_J=0.1,
        D=0.02,
        S0=0,
        Ncap=25,
        dt=0.02,
        n_steps=1000,
        seed=0,
        verbose=False,
        verbose_freq=1000,
    )
    n_runs: int = 500
    n_snaps: int = 3
    dim: int = 1
    periodic: bool = False


SIM = SimConfig()
ADOPT = AdoptConfig()
MEAN = MeanConfig()