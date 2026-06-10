from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import gammaln
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class SSSBFitParams:
    p: float
    q: float
    gamma_J: float
    k_J: float
    D: float
    S0: float
    r0: float
    r1: float = 0.0
    r2: float = 0.0
    r_max: float = 1e-4
    FI_a: float = 1.0
    FI_b: float = 1.0
    FI_c: float = 1e6


@dataclass(frozen=True)
class SSSBFitConfig:
    dt_years: float = 0.05
    eps_mu: float = 1e-12
    use_covariates: bool = True
    normalize_nll: bool = True
    capacity_link: str = "logistic"
    standardize_covariates: bool = True

    condition_on_seed_year: bool = False
    seed_year: int = 2007
    include_seed_likelihood: bool = True
    
    
def information_effect(I: np.ndarray, params: SSSBFitParams) -> np.ndarray:
    """
    F_I(I) = [I / (1 + a I)]^b * (1 - exp(-c I)).
    """
    I = np.maximum(np.asarray(I, dtype=float), 0.0)

    a = float(params.FI_a)
    b = float(params.FI_b)
    c = float(params.FI_c)

    if a <= 0 or b <= 0 or c <= 0:
        return np.full_like(I, np.nan)

    base = I / (1.0 + a * I)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = np.power(base, b) * (-np.expm1(-c * I))

    return np.maximum(out, 0.0)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def logistic(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)

    return out


def capacity_response(eta: np.ndarray, link: str) -> np.ndarray:
    link = str(link).lower()

    if link == "logistic":
        return logistic(eta)

    if link == "softplus":
        return softplus(eta)

    if link == "linear":
        return np.maximum(eta, 0.0)

    raise ValueError(f"Unknown capacity link: {link}")


def standardize_feature(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)

    out = np.zeros_like(x, dtype=float)
    if not np.any(good):
        return out

    mu = float(np.nanmean(x[good]))
    sd = float(np.nanstd(x[good]))

    if sd <= 0:
        return out

    out[good] = (x[good] - mu) / sd
    return out


def build_capacity(
    *,
    population: np.ndarray,
    pv_potential: np.ndarray,
    transmission_distance_km: np.ndarray,
    params: SSSBFitParams,
    use_covariates: bool,
    capacity_link: str = "logistic",
    standardize_covariates: bool = True,
) -> np.ndarray:
    population = np.asarray(population, dtype=float)
    population = np.clip(population, 0.0, None)

    eta = np.full_like(population, float(params.r0), dtype=float)

    if use_covariates:
        if standardize_covariates:
            z_pv = standardize_feature(pv_potential)
            z_grid = standardize_feature(transmission_distance_km)
        else:
            z_pv = np.asarray(pv_potential, dtype=float)
            z_grid = np.asarray(transmission_distance_km, dtype=float)

        eta += float(params.r1) * z_pv
        eta -= float(params.r2) * z_grid

    h = capacity_response(eta, capacity_link)

    if capacity_link in ("logistic",):
        K = population * float(params.r_max) * h

    elif capacity_link in ("softplus", "linear"):
        # Backward-compatible / experimental unbounded options.
        # Not recommended for stable simulations.
        K = population * h

    else:
        raise ValueError(f"Unknown capacity link: {capacity_link}")

    return np.clip(K, 0.0, None)


def observed_driven_nll(
    *,
    Y: np.ndarray,
    years: np.ndarray,
    population: np.ndarray,
    pv_potential: np.ndarray,
    transmission_distance_km: np.ndarray,
    L: csr_matrix,
    params: SSSBFitParams,
    cfg: SSSBFitConfig,
    return_details: bool = False,
):
    """
    Vectorized observation-driven Poisson NLL.

    Vectorized over nodes.
    Looped over years and substeps.
    """
    Y = np.asarray(Y, dtype=float)
    years = np.asarray(years, dtype=int)

    if Y.ndim != 2:
        raise ValueError("Y must be shaped (n_years, n_nodes).")

    n_years, n_nodes = Y.shape

    if n_years == 0:
        return 0.0 if not return_details else (0.0, {})

    K = build_capacity(
        population=population,
        pv_potential=pv_potential,
        transmission_distance_km=transmission_distance_km,
        params=params,
        use_covariates=cfg.use_covariates,
        capacity_link=cfg.capacity_link,
        standardize_covariates=cfg.standardize_covariates,
    )

    p = float(params.p)
    q = float(params.q)
    gamma_J = float(params.gamma_J)
    k_J = float(params.k_J)
    D = float(params.D)
    S0 = float(params.S0)
    
    r_max = float(params.r_max)

    FI_a = float(params.FI_a)
    FI_b = float(params.FI_b)
    FI_c = float(params.FI_c)
    
    if (
        p < 0 or q < 0 or gamma_J < 0 or k_J < 0 or D < 0 or S0 < 0
        or FI_a <= 0 or FI_b <= 0 or FI_c <= 0
        or r_max <= 0
    ):
        if return_details:
            return np.inf, {}
        return np.inf

    dt_req = float(cfg.dt_years)
    n_sub = int(round(1.0 / dt_req))
    if n_sub < 1:
        raise ValueError("dt_years is too large.")

    dt = 1.0 / n_sub

    I = np.zeros(n_nodes, dtype=float)
    J = np.zeros(n_nodes, dtype=float)
    W_cum = np.zeros(n_nodes, dtype=float)

    mu = np.zeros_like(Y, dtype=float)
    mu_U = np.zeros_like(Y, dtype=float)
    mu_V = np.zeros_like(Y, dtype=float)
    
    seed_idx = None
    if cfg.condition_on_seed_year:
        matches = np.where(years == int(cfg.seed_year))[0]
        if matches.size == 0:
            raise ValueError(f"seed_year={cfg.seed_year} not found in years.")
        seed_idx = int(matches[0])

    for yi in range(n_years):
        Y_year = Y[yi]
        
        if seed_idx is not None and yi == seed_idx:
            R = np.clip(K - W_cum, 0.0, None)
        
            # Seed-year events are treated as innovations.
            # They can still contribute to likelihood, forcing p*K to be plausible.
            if cfg.include_seed_likelihood:
                mu_seed = p * R
                mu_U[yi] = mu_seed
                mu_V[yi] = 0.0
                mu[yi] = mu_seed
        
            # But state evolution is conditioned on the observed seed events.
            jump = Y_year / n_sub
        
            for _ in range(n_sub):
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    J_plus = J + jump
                    I_new = I + dt * gamma_J * J_plus
                    LJ = L @ J_plus
                    J_new = J_plus + dt * (-k_J * J_plus + D * LJ + S0)
        
                if (
                    not np.all(np.isfinite(I_new))
                    or not np.all(np.isfinite(J_new))
                ):
                    if return_details:
                        return np.inf, {}
                    return np.inf
        
                I = np.maximum(I_new, 0.0)
                J = np.maximum(J_new, 0.0)
        
            W_cum += Y_year
            continue

        R = np.clip(K - W_cum, 0.0, None)
        jump = Y_year / n_sub

        for _ in range(n_sub):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                info_effect = information_effect(I, params)
                hazard_U = p
                hazard_V = q * info_effect
                hazard = hazard_U + hazard_V
                
                mu_U[yi] += R * hazard_U * dt
                mu_V[yi] += R * hazard_V * dt
                mu[yi] += R * hazard * dt

                J_plus = J + jump

                I_new = I + dt * gamma_J * J_plus

                LJ = L @ J_plus
                J_new = J_plus + dt * (-k_J * J_plus + D * LJ + S0)

            if (
                not np.all(np.isfinite(mu[yi]))
                or not np.all(np.isfinite(I_new))
                or not np.all(np.isfinite(J_new))
            ):
                if return_details:
                    return np.inf, {}
                return np.inf

            I = np.maximum(I_new, 0.0)
            J = np.maximum(J_new, 0.0)

        W_cum += Y_year

    eps = float(cfg.eps_mu)
    nll_terms = mu - Y * np.log(mu + eps) + gammaln(Y + 1.0)
    nll = float(np.sum(nll_terms))

    if cfg.normalize_nll:
        denom = max(float(np.sum(Y)), 1.0)
        nll /= denom

    if not return_details:
        return nll

    details = {
        "mu": mu,
        "capacity": K,
        "final_I": I,
        "final_J": J,
        "final_cumulative_observed": W_cum,
        "years": years,
        "dt_years_used": np.array([dt]),
        "n_substeps_per_year": np.array([n_sub]),
        "mu_U": mu_U,
        "mu_V": mu_V,
        "cum_mu_U": np.sum(mu_U, axis=0),
        "cum_mu_V": np.sum(mu_V, axis=0),
        "cum_mu_total": np.sum(mu, axis=0),
    }

    return nll, details