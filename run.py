from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from core import simulate_sssb, simulate_sssb_adoption_curve, simulate_sssb_selected_snapshots
from plotting import save_snapshots_2x2, save_adoption_curves, snapshot_indices, save_mean_1d_uv_overlay_timeseries_plots, save_mean_uv_overlay_timeseries_plots

from config import SIM, ADOPT, MEAN


def case_label(dim: int, periodic: bool, one_sided: bool) -> str:
    return f"dim{dim}_periodic{int(periodic)}_onesided{int(one_sided)}"


def curve_key(periodic: bool, one_sided: bool) -> str:
    return f"periodic{int(periodic)}_onesided{int(one_sided)}"


def usage_and_exit(code: int = 2) -> None:
    msg = (
        "Usage:\n"
        "  python run.py --all    Run everything (sim + adoption curves + mean)\n"
        "  python run.py --sim    Run full-field simulations + snapshot figures (12 cases)\n"
        "  python run.py --adopt  Run adoption curve simulations + curve figures/data (3 dims)\n"
        "  python run.py --mean Run 1D non-periodic mean study + figures\n"
        "\n"
        "Notes:\n"
        "  - All tunable parameters live in config.py.\n"
        "  - Exactly ONE flag must be provided.\n"
    )
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def parse_mode(argv: list[str]) -> str:
    valid = {"--all", "--sim", "--adopt", "--mean"}
    flags = [a for a in argv[1:] if a.startswith("--")]

    if len(flags) != 1 or flags[0] not in valid:
        usage_and_exit(2)

    return flags[0][2:]  # strip leading "--"


def run_sim() -> None:
    params = SIM.params
    n_snaps = SIM.n_snaps

    fig_root = Path("figures")
    fig_root.mkdir(exist_ok=True)

    setups = [(dim, periodic, one_sided)
              for dim in (1, 2, 3)
              for periodic in (False, True)
              for one_sided in (False, True)]
    total = len(setups)

    for idx, (dim, periodic, one_sided) in enumerate(setups, start=1):
        label = f"dim: {dim}, periodic: {periodic}, one-sided: {one_sided}"
        print(f"\n=== Running {label} ({idx}/{total}) ===")

        N_dim = params.grid_N[dim - 1]
        res = simulate_sssb(
            dim=dim,
            N=N_dim,
            periodic=periodic,
            one_sided=one_sided,
            params=params,
            _run_label=label,
            _run_idx=idx,
            _run_total=total,
        )

        out_dir = fig_root / case_label(dim, periodic, one_sided)
        save_snapshots_2x2(
            results=res,
            dim=dim,
            periodic=periodic,
            n_steps=params.n_steps,
            out_dir=out_dir,
            base_name=case_label(dim, periodic, one_sided),
            n_snaps=n_snaps,
        )

        U_final = res["U"][-1]
        V_final = res["V"][-1]
        print("Final total adopters:", int((U_final + V_final).sum()))

        del res


def run_adopt() -> None:
    params0 = ADOPT.params
    n_runs = int(getattr(ADOPT, "n_runs", 1))
    if n_runs <= 0:
        raise ValueError("ADOPT.n_runs must be a positive integer")

    fig_root = Path("figures")
    fig_root.mkdir(exist_ok=True)

    data_root = Path("curve_data")
    data_root.mkdir(exist_ok=True)

    for dim in (1, 2, 3):
        N_dim = params0.grid_N[dim - 1]
        setups = [(False, False), (False, True), (True, False), (True, True)]
        total = len(setups)

        curves: Dict[Tuple[bool, bool], Tuple[np.ndarray, np.ndarray]] = {}

        for idx, (periodic, one_sided) in enumerate(setups, start=1):
            label = f"dim: {dim}, periodic: {periodic}, one-sided: {one_sided}"
            print(f"\n=== Running {label} ({idx}/{total}) ===")

            A_sum: np.ndarray | None = None
            t_ref: np.ndarray | None = None

            # Monte Carlo average
            for r in range(n_runs):
                # vary seed per run; keep everything else identical
                params_r = params0.__class__(**{**params0.__dict__, "seed": params0.seed + r})

                out = simulate_sssb_adoption_curve(
                    dim=dim,
                    N=N_dim,
                    periodic=periodic,
                    one_sided=one_sided,
                    params=params_r,
                    _run_label=(label + (f", MC {r+1}/{n_runs}" if n_runs > 1 else "")),
                    _run_idx=idx,
                    _run_total=total,
                )

                t = out["t"]
                A = out["A"]

                if t_ref is None:
                    t_ref = t
                    A_sum = np.array(A, dtype=np.float64, copy=True)
                else:
                    # sanity check: all runs must match the same time grid
                    if t.shape != t_ref.shape or not np.all(t == t_ref):
                        raise ValueError("simulate_sssb_adoption_curve returned inconsistent t grids across runs")
                    A_sum += A.astype(np.float64)

            assert t_ref is not None and A_sum is not None
            A_mean = A_sum / float(n_runs)
            curves[(periodic, one_sided)] = (t_ref, A_mean)

        fig_path = fig_root / f"adoption_curves_dim{dim}.png"
        title = f"Cumulative adoption curves (dim={dim}, N={N_dim})"
        if n_runs > 1:
            title += f" — mean over {n_runs} runs"
        save_adoption_curves(
            curves=curves,
            dim=dim,
            out_path=fig_path,
            title=title,
        )

        npz_path = data_root / f"adoption_curves_dim{dim}.npz"
        np.savez_compressed(
            npz_path,
            **{f"t_{curve_key(p, o)}": curves[(p, o)][0] for (p, o) in curves},
            **{f"A_{curve_key(p, o)}": curves[(p, o)][1] for (p, o) in curves},
        )

        print(f"Saved: {fig_path}")
        print(f"Saved: {npz_path}")


def run_mean() -> None:
    params = MEAN.params
    n_runs = MEAN.n_runs
    n_snaps = MEAN.n_snaps
    dim = MEAN.dim
    periodic = MEAN.periodic

    if dim not in (1, 2):
        raise ValueError("MeanConfig currently supports only dim=1 or dim=2.")
    if periodic is True:
        raise ValueError("This mean study is intended for non-periodic only (periodic=False).")

    N_dim = params.grid_N[dim - 1]
    tidx = snapshot_indices(n_steps=params.n_steps, n_snaps=n_snaps)

    out_dir = Path("figures") / ("mean_1d" if dim == 1 else "mean_2d_collapsed")
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_first = 10  # print diagnostics for first 10 runs
    last_k = tidx.size - 1  # last snapshot index

    def collapse_if_needed(snaps_uv: np.ndarray) -> np.ndarray:
        """
        snaps_uv: (n_snaps, 2, *shape)
          dim=1 -> (n_snaps, 2, N)
          dim=2 -> (n_snaps, 2, N, N)

        Returns:
          dim=1 -> same shape (n_snaps, 2, N)
          dim=2 -> collapsed along y (axis=2) => (n_snaps, 2, N_x)
                  Here we interpret:
                    field[y, x] with y = axis 0, x = axis 1
                  so average over y => mean over axis=2 after (n_snaps,2,y,x) layout.
        """
        if dim == 1:
            return snaps_uv
        # snaps_uv shape: (n_snaps, 2, N, N) where axes are (y, x) in the last two dims
        return snaps_uv.mean(axis=2)  # average over y -> (n_snaps, 2, N_x)

    def run_mean_uv(one_sided: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean_U, mean_V) each shape (n_snaps, N_dim_collapsed),
        computed via streaming mean.
        For dim=1: N_dim_collapsed = N_dim
        For dim=2: N_dim_collapsed = N_dim (x-axis length), after collapsing y.
        """
        sum_U = np.zeros((tidx.size, N_dim), dtype=np.float64)
        sum_V = np.zeros((tidx.size, N_dim), dtype=np.float64)

        base_kwargs = {**params.__dict__, "verbose": False}

        for r in range(n_runs):
            seed_r = params.seed + r
            params_r = params.__class__(**{**base_kwargs, "seed": seed_r})

            snaps = simulate_sssb_selected_snapshots(
                dim=dim,
                N=N_dim,
                periodic=periodic,
                one_sided=one_sided,
                params=params_r,
                snapshot_tidx=tidx,
                field="UV",
            ).astype(np.float64)  # (n_snaps, 2, ...)

            snaps = collapse_if_needed(snaps)  # now (n_snaps, 2, N_dim)

            # --- DEBUG: show first 10 nodes at the last snapshot time for first 10 runs
            if r < debug_first:
                u10 = snaps[last_k, 0, :10]
                v10 = snaps[last_k, 1, :10]
                print(
                    f"[DEBUG mean] dim={dim} one_sided={one_sided} run={r+1:02d}/{n_runs} seed={seed_r} "
                    f"t={int(tidx[last_k])} U[:10]={np.array2string(u10, precision=3)} "
                    f"V[:10]={np.array2string(v10, precision=3)}"
                )

            sum_U += snaps[:, 0, :]
            sum_V += snaps[:, 1, :]

            if getattr(params, "verbose", False) and (r == 0 or (r + 1) % 50 == 0 or (r + 1) == n_runs):
                print(f"  mean study: dim={dim}, one_sided={one_sided} --- run {r+1}/{n_runs}")

        mean_U = sum_U / float(n_runs)
        mean_V = sum_V / float(n_runs)
        return mean_U, mean_V

    print(f"\n=== Computing means for dim={dim} non-periodic ===")

    mean_U_twodir, mean_V_twodir = run_mean_uv(one_sided=False)
    mean_U_onedir, mean_V_onedir = run_mean_uv(one_sided=True)

    # Save exactly n_snaps figures total (one per timestep),
    # each with two panels (U top, V bottom) and both directionality curves.
    save_mean_uv_overlay_timeseries_plots(
        mean_U_twodir=mean_U_twodir,
        mean_U_onedir=mean_U_onedir,
        mean_V_twodir=mean_V_twodir,
        mean_V_onedir=mean_V_onedir,
        tidx=tidx,
        out_dir=out_dir,
        base_name="mean_values_overlay",
        title=("Mean values" + ("" if dim == 1 else " (2D collapsed over y)")),
    )

    print("Saved mean overlay plots to:", out_dir)


def main() -> None:
    mode = parse_mode(sys.argv)

    if mode == "sim":
        run_sim()
        return
    if mode == "adopt":
        run_adopt()
        return
    if mode == "mean":
        run_mean()
        return
    if mode == "all":
        run_sim()
        run_adopt()
        run_mean()
        return

    usage_and_exit(2)


if __name__ == "__main__":
    main()