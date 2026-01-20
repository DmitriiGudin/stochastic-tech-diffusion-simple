from __future__ import annotations

from pathlib import Path
import math
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def snapshot_indices(n_steps: int, n_snaps: int) -> np.ndarray:
    if n_snaps <= 0:
        raise ValueError("n_snaps must be positive")
    T = int(n_steps)
    idx = np.linspace(0, T, n_snaps + 1)[1:]
    idx = np.rint(idx).astype(int)
    idx = np.clip(idx, 0, T)
    idx = np.maximum.accumulate(idx)
    return idx


def _collapse_to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.sum(axis=0)
    raise ValueError(f"Unsupported field ndim={arr.ndim}")


def _coords_1d_circle(n: int) -> Tuple[np.ndarray, np.ndarray]:
    # Place points on a unit circle; data-units consistent for patch radius
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return x, y


def _coords_1d_snake(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D "snake" with 3 vertical legs and 2 horizontal connectors:

      start top-left
        ↓ (left leg)
        → (to middle)
        ↑ (middle leg)
        → (to right)
        ↓ (right leg)

    We choose the horizontal span to scale with the vertical height so
    the curve fills the plot width instead of staying narrow.
    """
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Choose a reasonably tall height first.
    # We'll make width scale with height.
    H = int(math.ceil(math.sqrt(n)))  # heuristic: height ~ sqrt(n)
    # Make total width comparable to height (fills x-axis visually)
    W = max(2, int(round(H)))  # overall width ~ 0.75*H (tweakable)

    # Split width into two connectors (left->middle, middle->right)
    W1 = W
    W2 = W

    # Ensure the path has at least n points.
    # Total points = 1 + 3*H + (W1 + W2)
    # If still too short, increase H a bit.
    total_len = 1 + 3 * H + (W1 + W2)
    if total_len < n:
        extra = n - total_len
        # Each +1 of H adds 3 points
        H += int(math.ceil(extra / 3.0))

    pts: list[tuple[float, float]] = []
    x, y = 0, 0
    pts.append((x, y))

    def step(dx: int, dy: int, k: int) -> None:
        nonlocal x, y
        for _ in range(k):
            x += dx
            y += dy
            pts.append((x, y))

    # Down left leg
    step(0, 1, H)
    # Right to middle
    step(1, 0, W1)
    # Up middle leg
    step(0, -1, H)
    # Right to right edge
    step(1, 0, W2)
    # Down right leg
    step(0, 1, H)

    pts = pts[:n]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    # Flip vertically so it visually starts at "top-left"
    ys = -ys

    return xs, ys


def _draw_circles(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    radius: float,
    periodic_lr_edges: bool,
    periodic_lr_edge_xmin: float,
    periodic_lr_edge_xmax: float,
) -> None:
    values = values.astype(float)
    vmin = float(np.nanmin(values)) if values.size else 0.0
    vmax = float(np.nanmax(values)) if values.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0  # avoid zero-range normalization

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap()  # default colormap

    patches = [Circle((xi, yi), radius=radius) for xi, yi in zip(x, y)]
    pc = PatchCollection(patches, array=values, cmap=cmap, norm=norm, linewidths=0.0)
    ax.add_collection(pc)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Tight bounds
    pad = radius * 1.2
    ax.set_xlim(float(np.min(x)) - pad, float(np.max(x)) + pad)
    ax.set_ylim(float(np.min(y)) - pad, float(np.max(y)) + pad)

    # Periodic markers (left/right)
    if periodic_lr_edges:
        ax.plot([periodic_lr_edge_xmin, periodic_lr_edge_xmin], ax.get_ylim(), color="red", linewidth=2.5)
        ax.plot([periodic_lr_edge_xmax, periodic_lr_edge_xmax], ax.get_ylim(), color="red", linewidth=2.5)

    # Colorbar per panel
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)


def _plot_field_panel(
    ax: plt.Axes,
    field: np.ndarray,
    dim: int,
    periodic: bool,
    title: str,
) -> None:
    """
    - 1D: periodic -> circle; non-periodic -> snake
    - 2D: circles on grid, no mesh
    - 3D: collapse -> 2D circles on grid, no mesh
    - “touching”: set radius ~ 0.5 for unit grid spacing; for circle use based on chord length
    - periodic edges: for 2D/3D periodic, show left/right red boundary lines
    """
    if dim == 1:
        z = field.astype(float).reshape(-1)
        n = z.size
        if periodic:
            x, y = _coords_1d_circle(n)
            # chord length between neighbors on unit circle: 2 sin(pi/n)
            # choose radius to ~touch but not overlap too much
            chord = 2.0 * math.sin(math.pi / max(n, 2))
            radius = 0.48 * chord
        else:
            x, y = _coords_1d_snake(n)
            radius = 0.48  # unit grid spacing in snake coords
        _draw_circles(
            ax, x, y, z, title,
            radius=radius,
            periodic_lr_edges=False,
            periodic_lr_edge_xmin=0.0,
            periodic_lr_edge_xmax=0.0,
        )
        return

    # 2D or collapsed 3D
    Z2 = _collapse_to_2d(field.astype(float))
    Ny, Nx = Z2.shape
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    x = xx.ravel().astype(float)
    y = yy.ravel().astype(float)
    vals = Z2.ravel()

    radius = 0.48  # unit grid spacing => touching circles

    # For periodic cases: show periodicity on left/right only (as requested).
    # We draw red lines at x = -0.5 and x = (Nx-1)+0.5.
    periodic_lr_edges = bool(periodic)
    edge_xmin = -0.5
    edge_xmax = (Nx - 1) + 0.5

    _draw_circles(
        ax, x, y, vals, title,
        radius=radius,
        periodic_lr_edges=periodic_lr_edges,
        periodic_lr_edge_xmin=edge_xmin,
        periodic_lr_edge_xmax=edge_xmax,
    )


def save_snapshots_2x2(
    results: Dict[str, np.ndarray],
    dim: int,
    periodic: bool,
    n_steps: int,
    out_dir: Path,
    base_name: str,
    n_snaps: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    U = results["U"]
    V = results["V"]
    I = results["I"]
    J = results["J"]

    idxs = snapshot_indices(n_steps=n_steps, n_snaps=n_snaps)

    for k, t_idx in enumerate(idxs, start=1):
        U_t = U[t_idx]
        V_t = V[t_idx]
        I_t = I[t_idx]
        J_t = J[t_idx]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        axU, axV = axes[0, 0], axes[0, 1]
        axI, axJ = axes[1, 0], axes[1, 1]

        _plot_field_panel(axU, U_t, dim, periodic, f"U (iteration {t_idx} / {n_steps})")
        _plot_field_panel(axV, V_t, dim, periodic, f"V (iteration {t_idx} / {n_steps})")
        _plot_field_panel(axI, I_t, dim, periodic, f"I (iteration {t_idx} / {n_steps})")
        _plot_field_panel(axJ, J_t, dim, periodic, f"J (iteration {t_idx} / {n_steps})")

        fname = out_dir / f"{base_name}_snap{k:02d}_t{t_idx:05d}.png"
        fig.savefig(fname, dpi=160)
        plt.close(fig)
        
        
def pretty_label(periodic: bool, one_sided: bool) -> str:
    per = "Periodic" if periodic else "Non-periodic"
    diff = "One-directional" if one_sided else "Two-directional"
    return f"{per}, {diff}"
        
        
def save_adoption_curves(
    curves: Dict[Tuple[bool, bool], Tuple[np.ndarray, np.ndarray]],
    *,
    dim: int,
    out_path: Path,
    title: str | None = None,
) -> None:
    """
    curves[(periodic, one_sided)] = (t_idx, adoption_fraction)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for (periodic, one_sided), (t, A) in curves.items():
        ax.plot(t, A, label=pretty_label(periodic, one_sided))

    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative adoption fraction")
    ax.set_xlim(0, max(int(np.max(next(iter(curves.values()))[0])), 1))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title is None:
        title = f"Cumulative adoption curves (dim={dim})"
    ax.set_title(title)

    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    
    
def save_mean_1d_uv_overlay_timeseries_plots(
    mean_U_twodir: np.ndarray,   # (n_snaps, N)
    mean_U_onedir: np.ndarray,   # (n_snaps, N)
    mean_V_twodir: np.ndarray,   # (n_snaps, N)
    mean_V_onedir: np.ndarray,   # (n_snaps, N)
    tidx: np.ndarray,            # (n_snaps,)
    out_dir: Path,
    *,
    base_name: str = "mean_values_overlay",
    title: str = "Mean values",
) -> None:
    """
    Saves one figure per snapshot time (so n_snaps figures total).
    Each figure: two panels (U top, V bottom), x=Node ID, y=Value.
    Each panel overlays two curves:
      - Two-directional (one_sided=False)
      - One-directional (one_sided=True)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basic checks
    if mean_U_twodir.shape != mean_U_onedir.shape:
        raise ValueError("mean_U_twodir and mean_U_onedir must match shapes")
    if mean_V_twodir.shape != mean_V_onedir.shape:
        raise ValueError("mean_V_twodir and mean_V_onedir must match shapes")
    if mean_U_twodir.shape != mean_V_twodir.shape:
        raise ValueError("U and V mean arrays must match shapes")

    if mean_U_twodir.ndim != 2:
        raise ValueError("Mean arrays must be 2D: (n_snaps, N)")
    if tidx.shape[0] != mean_U_twodir.shape[0]:
        raise ValueError("tidx length must match n_snaps")

    n_snaps, N = mean_U_twodir.shape
    x = np.arange(N, dtype=int)

    for k in range(n_snaps):
        t = int(tidx[k])

        fig, axes = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
        )
        axU, axV = axes[0], axes[1]

        # U panel
        axU.plot(x, mean_U_twodir[k], label="U (two-directional)")
        axU.plot(x, mean_U_onedir[k], label="U (one-directional)")
        axU.set_ylabel("Value")
        axU.legend()

        # V panel
        axV.plot(x, mean_V_twodir[k], label="V (two-directional)")
        axV.plot(x, mean_V_onedir[k], label="V (one-directional)")
        axV.set_ylabel("Value")
        axV.set_xlabel("Node ID")
        axV.legend()

        fig.suptitle(f"{title} (iteration {t})")

        fname = out_dir / f"{base_name}_t{t:05d}.png"
        fig.savefig(fname, dpi=170)
        plt.close(fig)