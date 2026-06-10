from __future__ import annotations


DEFAULT = {
    "paths": {
        "admin1_shp": "data/raw/ne_10m_admin_1_states_provinces_lakes/ne_10m_admin_1_states_provinces_lakes.shp",
        "county_shp": "data/raw/cb_2023_us_county_5m/cb_2023_us_county_5m.shp",
        "pop_csv": "data/raw/zip_population_all.csv",
        "lspv_csv": "data/raw/uspvdb_v4_0_20260414.csv",
        "transmission_shp": "data/raw/Transmission_Lines/Transmission_Lines.shp",
        "pvout_tif": "data/raw/PVOUT.tif",
    },

    "region": {
        "states": [],
        "counties": [],
    },

    "mesh": {
        "h_km": 25.0,
        "simplify_km": 5.0,
        "epsg_project": 5070,
        "overwrite_mesh": False,
    },

    "density": {
        "smooth_length_km": 75,
        "smooth_k_neighbors": 100,
        "smooth_kernel": "gaussian",
        "transmission_buffer_km": 100,
        "plot_pop_year": 2020,
        "adoption_plot_scale": "log1p",  # "linear", "log1p" or "mixed"
        "mixed_log_range": 8.0,
        "adoption_plot_quantile": 1.0,  # use 1.0 for true max; 0.99 for clipped scale
        "field_plot_quantile": 1.0,  # use 1.0 for true max; 0.99 for clipped scale
        "adoption_shared_colorbar": True,
    },

    "fit": {
        "dt_years": 0.05,
        "use_covariates": True,
        "fit_S0": False,
        "n_random": 5000,
        "maxiter": 500,
        "seed": 1337,
        "population_key": "population_smooth_2020",
        "progress_freq": 10,
        "year_window": None,  # e.g. [2007, 2025], inclusive
        "condition_on_seed_year": True,
        "seed_year": 2007, # condition on all observed years from start year through this year
        "include_seed_likelihood": True,
    },
    
    "capacity": {
        "link": "logistic",          # "logistic" or "linear"
        "standardize_covariates": True,
    },

    "param_bounds": {
        "p": [1e-9, 1e-4],
        "q": [1e-5, 0.1],
        "gamma_J": [1e-3, 1000],
        "k_J": [1e-6, 1e-3],
        "D": [1e-12, 1],
        "S0": [0, 0],
        "r_max": [1e-6, 1e-2],
        "r0": [0.01, 100],
        "r1": [0.01, 100],
        "r2": [0.01, 100],
        "FI_a": [0.01, 100],
        "FI_b": [0.01, 10],
        "FI_c": [0.1, 100],
    },

    "initial": {
        "p": 1e-4,
        "q": 1e-1,
        "gamma_J": 1e-1,
        "k_J": 0.01,
        "D": 100,
        "S0": 0,
        "r_max": 1e-4,
        "r0": 1e-5,
        "r1": 1e-9,
        "r2": 1e-6,
        "FI_a": 1,
        "FI_b": 1,
        "FI_c": 1,
    },
    
    "simulation": {
        "forecast_year": 2050,
        "seed": 1337,
        "fps": 2,
        "n_single_runs": 10,
        "run_batch": True,
        "batch_n_sims": 100,
    },
}


CONFIGS = {
    "default": {},
    
    "CA_NV_AZ_UT_config": {
        "region": {
            "states": ["CA", "NV", "AZ", "UT"],
            "counties": [],
        },
        "mesh": {
            "h_km": 6,
            "simplify_km": 18,
        },
        "fit": {
            "year_window": [2007, 2025],
            "seed_year": 2016,
        },
    },
    
    "MD_DC_VA_DE_config": {
        "region": {
            "states": ["MD", "DC", "VA", "DE"],
            "counties": [],
        },
        "mesh": {
            "h_km": 2,
            "simplify_km": 6,
        },
        "fit": {
            "year_window": [2011, 2025],
            "seed_year": 2018,
        },
    },
    
    "IN_OH_IL_MI_WI_MO_MN_IA_config": {
        "region": {
            "states": ['IN', 'OH', 'IL', 'MI', 'WI', 'MO', 'MN', 'IA'],
            "counties": [],
        },
        "mesh": {
            "h_km": 7,
            "simplify_km": 21,
        },
        "fit": {
            "year_window": [2011, 2025],
            "seed_year": 2018,
        },
    },
    
    "TX_config": {
        "region": {
            "states": ['TX'],
            "counties": [],
        },
        "mesh": {
            "h_km": 5,
            "simplify_km": 15,
        },
        "fit": {
            "year_window": [2015, 2025],
            "seed_year": 2020,
        },
    },
    
    "FL_GA_SC_NC_config": {
        "region": {
            "states": ['FL', 'GA', 'SC', 'NC'],
            "counties": [],
        },
        "mesh": {
            "h_km": 5,
            "simplify_km": 15,
        },
        "fit": {
            "year_window": [2011, 2025],
            "seed_year": 2018,
        },
    },
}