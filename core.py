from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import time


@dataclass(frozen=True)
class SSSBParams:
    grid_N: tuple[int, int, int] = (40, 40, 20)
    
    p: float = 0.01
    q: float = 0.3
    gamma_J: float = 0.1
    k_J: float = 0.5
    D: float = 0.1
    S0: float = 0
    Ncap: int = 10
    dt: float = 0.02
    n_steps: int = 2000
    seed: int = 0

    verbose: bool = True
    verbose_freq: int = 1000


def _fmt_hhmmss(seconds: float) -> str:
    s = int(seconds)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _laplacian_full_reflecting(J: np.ndarray) -> np.ndarray:
    ndim = J.ndim
    if ndim == 1:
        P = np.pad(J, pad_width=1, mode="edge")
        return (P[:-2] + P[2:]) - 2.0 * J
    if ndim == 2:
        P = np.pad(J, pad_width=((1, 1), (1, 1)), mode="edge")
        up = P[:-2, 1:-1]
        down = P[2:, 1:-1]
        left = P[1:-1, :-2]
        right = P[1:-1, 2:]
        return (up + down + left + right) - 4.0 * J
    if ndim == 3:
        P = np.pad(J, pad_width=((1, 1), (1, 1), (1, 1)), mode="edge")
        xm = P[1:-1, 1:-1, :-2]
        xp = P[1:-1, 1:-1, 2:]
        ym = P[1:-1, :-2, 1:-1]
        yp = P[1:-1, 2:, 1:-1]
        zm = P[:-2, 1:-1, 1:-1]
        zp = P[2:, 1:-1, 1:-1]
        return (xm + xp + ym + yp + zm + zp) - 6.0 * J
    raise ValueError(f"Unsupported ndim={ndim}")


def _laplacian_full_periodic(J: np.ndarray) -> np.ndarray:
    L = np.zeros_like(J, dtype=float)
    for axis in range(J.ndim):
        L += np.roll(J, +1, axis=axis) + np.roll(J, -1, axis=axis) - 2.0 * J
    return L


def _laplacian_half_reflecting(J: np.ndarray) -> np.ndarray:
    ndim = J.ndim
    if ndim == 1:
        P = np.pad(J, pad_width=1, mode="edge")
        plus = P[2:]
        return plus - J
    if ndim == 2:
        P = np.pad(J, pad_width=((1, 1), (1, 1)), mode="edge")
        plus_x = P[2:, 1:-1]
        plus_y = P[1:-1, 2:]
        return (plus_x - J) + (plus_y - J)
    if ndim == 3:
        P = np.pad(J, pad_width=((1, 1), (1, 1), (1, 1)), mode="edge")
        plus_x = P[1:-1, 1:-1, 2:]
        plus_y = P[1:-1, 2:, 1:-1]
        plus_z = P[2:, 1:-1, 1:-1]
        return (plus_x - J) + (plus_y - J) + (plus_z - J)
    raise ValueError(f"Unsupported ndim={ndim}")


def _laplacian_half_periodic(J: np.ndarray) -> np.ndarray:
    L = np.zeros_like(J, dtype=float)
    for axis in range(J.ndim):
        L += np.roll(J, -1, axis=axis) - J
    return L


def simulate_sssb(
    dim: int,
    N: int,
    periodic: bool,
    one_sided: bool,
    params: SSSBParams,
    *,
    _run_label: str = "",
    _run_idx: int = 1,
    _run_total: int = 1,
    _t0_wall: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate the SSSB model on:
      - 1D grid: (N,)
      - 2D grid: (N,N)
      - 3D grid: (N,N,N)

    Returns dict with arrays:
      U[t,...], V[t,...] integers
      I[t,...], J[t,...] floats
    """
    if dim not in (1, 2, 3):
        raise ValueError("dim must be 1, 2, or 3")

    if params.verbose_freq <= 0:
        raise ValueError("verbose_freq must be positive")

    shape = (N,) * dim
    rng = np.random.default_rng(params.seed)

    U = np.zeros(shape, dtype=np.int64)
    V = np.zeros(shape, dtype=np.int64)
    I = np.zeros(shape, dtype=np.float64)
    J = np.zeros(shape, dtype=np.float64)

    T = params.n_steps
    U_hist = np.zeros((T + 1, *shape), dtype=np.int64)
    V_hist = np.zeros((T + 1, *shape), dtype=np.int64)
    I_hist = np.zeros((T + 1, *shape), dtype=np.float64)
    J_hist = np.zeros((T + 1, *shape), dtype=np.float64)

    U_hist[0] = U
    V_hist[0] = V
    I_hist[0] = I
    J_hist[0] = J

    if periodic and not one_sided:
        lap = _laplacian_full_periodic
    elif periodic and one_sided:
        lap = _laplacian_half_periodic
    elif (not periodic) and (not one_sided):
        lap = _laplacian_full_reflecting
    else:
        lap = _laplacian_half_reflecting

    D_eff = (2.0 * params.D) if one_sided else params.D

    dt = float(params.dt)
    p = float(params.p)
    q = float(params.q)
    gamma_J = float(params.gamma_J)
    k_J = float(params.k_J)
    S0 = float(params.S0)
    Ncap = int(params.Ncap)

    if _t0_wall is None:
        _t0_wall = time.time()

    for t in range(1, T + 1):
        if params.verbose and (t == 1 or (t % params.verbose_freq == 0) or (t == T)):
            elapsed = time.time() - _t0_wall
            stamp = _fmt_hhmmss(elapsed)
            label = _run_label or f"dim: {dim}, periodic: {periodic}, one-sided: {one_sided}"
            print(f"[{stamp}] {label} ({_run_idx}/{_run_total}) --- Iteration {t} / {T}")

        R = Ncap - U - V
        R = np.maximum(R, 0)

        sI = I / (1.0 + I)
        a = p + q * sI

        adopt_prob = 1.0 - np.exp(-a * dt)
        total_adopt = rng.binomial(R, adopt_prob)

        innov_prob = np.where(a > 0.0, p / a, 1.0)
        dU = rng.binomial(total_adopt, innov_prob)
        dV = total_adopt - dU

        U = U + dU
        V = V + dV

        J = J + total_adopt.astype(np.float64)

        I = I + dt * (gamma_J * J)
        J = J + dt * (-k_J * J + D_eff * lap(J) + S0)

        I = np.maximum(I, 0.0)
        J = np.maximum(J, 0.0)

        U_hist[t] = U
        V_hist[t] = V
        I_hist[t] = I
        J_hist[t] = J

    return {"U": U_hist, "V": V_hist, "I": I_hist, "J": J_hist}


def run_all_12_cases(
    N: int,
    params: SSSBParams,
) -> Dict[Tuple[int, bool, bool], Dict[str, np.ndarray]]:
    """
    Returns a dict keyed by (dim, periodic, one_sided).
    """
    setups = [(dim, periodic, one_sided)
              for dim in (1, 2, 3)
              for periodic in (False, True)
              for one_sided in (False, True)]

    out: Dict[Tuple[int, bool, bool], Dict[str, np.ndarray]] = {}
    t0 = time.time()
    total = len(setups)

    for idx, (dim, periodic, one_sided) in enumerate(setups, start=1):
        label = f"dim: {dim}, periodic: {periodic}, one-sided: {one_sided}"
        key = (dim, periodic, one_sided)
        out[key] = simulate_sssb(
            dim=dim,
            N=N,
            periodic=periodic,
            one_sided=one_sided,
            params=params,
            _run_label=label,
            _run_idx=idx,
            _run_total=total,
            _t0_wall=t0,
        )

    return out


def simulate_sssb_adoption_curve(
    dim: int,
    N: int,
    periodic: bool,
    one_sided: bool,
    params: SSSBParams,
    *,
    _run_label: str = "",
    _run_idx: int = 1,
    _run_total: int = 1,
    _t0_wall: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Memory-light run: returns only the cumulative adoption fraction curve.

    Returns:
      {"t": t_idx array (T+1,),
       "A": adoption fraction array (T+1,)}
    where A(t) = sum(U+V) / (Ncap * num_nodes) ∈ [0,1].
    """
    if dim not in (1, 2, 3):
        raise ValueError("dim must be 1, 2, or 3")

    if params.verbose_freq <= 0:
        raise ValueError("verbose_freq must be positive")

    shape = (N,) * dim
    rng = np.random.default_rng(params.seed)

    U = np.zeros(shape, dtype=np.int64)
    V = np.zeros(shape, dtype=np.int64)
    I = np.zeros(shape, dtype=np.float64)
    J = np.zeros(shape, dtype=np.float64)

    T = int(params.n_steps)

    if periodic and not one_sided:
        lap = _laplacian_full_periodic
    elif periodic and one_sided:
        lap = _laplacian_half_periodic
    elif (not periodic) and (not one_sided):
        lap = _laplacian_full_reflecting
    else:
        lap = _laplacian_half_reflecting

    D_eff = (2.0 * params.D) if one_sided else params.D

    dt = float(params.dt)
    p = float(params.p)
    q = float(params.q)
    gamma_J = float(params.gamma_J)
    k_J = float(params.k_J)
    S0 = float(params.S0)
    Ncap = int(params.Ncap)

    if _t0_wall is None:
        _t0_wall = time.time()

    num_nodes = int(np.prod(shape))
    total_capacity = float(Ncap * num_nodes)

    t_idx = np.arange(T + 1, dtype=int)
    A = np.zeros(T + 1, dtype=np.float64)

    # initial adoption fraction
    A[0] = float((U + V).sum()) / total_capacity

    for t in range(1, T + 1):
        if params.verbose and (t == 1 or (t % params.verbose_freq == 0) or (t == T)):
            elapsed = time.time() - _t0_wall
            stamp = _fmt_hhmmss(elapsed)
            label = _run_label or f"dim: {dim}, periodic: {periodic}, one-sided: {one_sided}"
            print(f"[{stamp}] {label} ({_run_idx}/{_run_total}) --- Iteration {t} / {T}")

        R = Ncap - U - V
        R = np.maximum(R, 0)

        sI = I / (1.0 + I)
        a = p + q * sI

        adopt_prob = 1.0 - np.exp(-a * dt)
        total_adopt = rng.binomial(R, adopt_prob)

        innov_prob = np.where(a > 0.0, p / a, 1.0)
        dU = rng.binomial(total_adopt, innov_prob)
        dV = total_adopt - dU

        U = U + dU
        V = V + dV

        # adoption adds to J immediately
        J = J + total_adopt.astype(np.float64)

        # flow between jumps
        I = I + dt * (gamma_J * J)
        J = J + dt * (-k_J * J + D_eff * lap(J) + S0)

        I = np.maximum(I, 0.0)
        J = np.maximum(J, 0.0)

        A[t] = float((U + V).sum()) / total_capacity

    return {"t": t_idx, "A": A}


def simulate_sssb_selected_snapshots(
    dim: int,
    N: int,
    periodic: bool,
    one_sided: bool,
    params: SSSBParams,
    snapshot_tidx: np.ndarray,
    *,
    field: str = "U",  # "U","V","I","J","UV","UI","VJ", etc.
    _run_label: str = "",
    _run_idx: int = 1,
    _run_total: int = 1,
    _t0_wall: Optional[float] = None,
) -> np.ndarray:
    """
    Runs the SSSB model but only returns selected field(s) at specified timestep indices.

    If `field` is a single letter (e.g. "U"), returns snaps[k, ...].
    If `field` contains multiple letters (e.g. "UV"), returns snaps[k, f, ...]
      where f indexes the requested fields in the order given.
    """
    if dim not in (1, 2, 3):
        raise ValueError("dim must be 1, 2, or 3")

    snapshot_tidx = np.asarray(snapshot_tidx, dtype=int)
    if snapshot_tidx.ndim != 1 or snapshot_tidx.size == 0:
        raise ValueError("snapshot_tidx must be a non-empty 1D array")
    if np.any(snapshot_tidx < 0) or np.any(snapshot_tidx > params.n_steps):
        raise ValueError("snapshot_tidx contains out-of-range indices")

    field = field.upper()
    allowed = {"U", "V", "I", "J"}
    fields = list(field)  # allows "UV" etc.
    if any(f not in allowed for f in fields):
        raise ValueError("field must consist of letters from {U,V,I,J}, e.g. 'U' or 'UV'")

    shape = (N,) * dim
    rng = np.random.default_rng(params.seed)

    U = np.zeros(shape, dtype=np.int64)
    V = np.zeros(shape, dtype=np.int64)
    I = np.zeros(shape, dtype=np.float64)
    J = np.zeros(shape, dtype=np.float64)

    if periodic and not one_sided:
        lap = _laplacian_full_periodic
    elif periodic and one_sided:
        lap = _laplacian_half_periodic
    elif (not periodic) and (not one_sided):
        lap = _laplacian_full_reflecting
    else:
        lap = _laplacian_half_reflecting

    D_eff = (2.0 * params.D) if one_sided else params.D

    dt = float(params.dt)
    p = float(params.p)
    q = float(params.q)
    gamma_J = float(params.gamma_J)
    k_J = float(params.k_J)
    S0 = float(params.S0)
    Ncap = int(params.Ncap)

    if _t0_wall is None:
        _t0_wall = time.time()

    order = np.argsort(snapshot_tidx)
    tidx_sorted = snapshot_tidx[order]

    # snaps shape depends on number of requested fields
    if len(fields) == 1:
        snaps = np.zeros((tidx_sorted.size, *shape), dtype=np.float64)
    else:
        snaps = np.zeros((tidx_sorted.size, len(fields), *shape), dtype=np.float64)

    def pack_current() -> np.ndarray:
        cur_map = {"U": U, "V": V, "I": I, "J": J}
        if len(fields) == 1:
            return cur_map[fields[0]].astype(np.float64)
        return np.stack([cur_map[f].astype(np.float64) for f in fields], axis=0)

    snap_ptr = 0

    if tidx_sorted[0] == 0:
        snaps[snap_ptr] = pack_current()
        snap_ptr += 1

    T = int(params.n_steps)

    for t in range(1, T + 1):
        if params.verbose and (t == 1 or (t % params.verbose_freq == 0) or (t == T)):
            elapsed = time.time() - _t0_wall
            stamp = _fmt_hhmmss(elapsed)
            label = _run_label or f"dim: {dim}, periodic: {periodic}, one-sided: {one_sided}"
            print(f"[{stamp}] {label} ({_run_idx}/{_run_total}) --- Iteration {t} / {T}")

        R = Ncap - U - V
        R = np.maximum(R, 0)

        sI = I / (1.0 + I)
        a = p + q * sI

        adopt_prob = 1.0 - np.exp(-a * dt)
        total_adopt = rng.binomial(R, adopt_prob)

        innov_prob = np.where(a > 0.0, p / a, 1.0)
        dU = rng.binomial(total_adopt, innov_prob)
        dV = total_adopt - dU

        U = U + dU
        V = V + dV

        J = J + total_adopt.astype(np.float64)

        I = I + dt * (gamma_J * J)
        J = J + dt * (-k_J * J + D_eff * lap(J) + S0)

        I = np.maximum(I, 0.0)
        J = np.maximum(J, 0.0)

        while snap_ptr < tidx_sorted.size and t == tidx_sorted[snap_ptr]:
            snaps[snap_ptr] = pack_current()
            snap_ptr += 1

        if snap_ptr >= tidx_sorted.size:
            break

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return snaps[inv]