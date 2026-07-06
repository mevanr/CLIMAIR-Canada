"""
CLIMAIR-Canada proof-of-concept simulation

This script creates a scientifically structured proof-of-concept for CLIMAIR-Canada.

Author: Mevan Rajakaruna, Harshana Rajakaruna
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 42
    n_regions: int = 18
    start: str = "2018-01-01"
    end: str = "2024-12-31"
    freq: str = "W-MON"
    output_dir: str = "climair_poc_outputs"
    n_bootstrap_models: int = 6


CFG = Config()
rng = np.random.default_rng(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Synthetic Canadian geography and covariates
# -----------------------------------------------------------------------------

def make_regions(n_regions: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create synthetic Canadian communities across broad eco-climate regions."""
    ecozones = np.array(["Pacific", "Prairie", "Boreal", "Atlantic", "Northern", "UrbanCorridor"])
    eco = rng.choice(ecozones, size=n_regions, p=[0.14, 0.18, 0.24, 0.14, 0.12, 0.18])

    # Synthetic projected coordinates, not real lat/lon.
    x = rng.uniform(0, 1000, n_regions)
    y = rng.uniform(0, 700, n_regions)

    # Urbanization and vulnerability are region-level attributes.
    urban_index = np.clip(
        rng.beta(2, 4, n_regions) + 0.45 * (eco == "UrbanCorridor") + 0.15 * (eco == "Atlantic"), 0, 1
    )
    road_density = np.clip(0.25 + 1.6 * urban_index + rng.normal(0, 0.18, n_regions), 0.05, 2.4)
    green_space = np.clip(0.75 - 0.55 * urban_index + rng.normal(0, 0.08, n_regions), 0.05, 0.95)
    pop = np.exp(rng.normal(10.6 + 1.2 * urban_index, 0.65)).astype(int)
    vulnerability = np.clip(
        0.25 + 0.35 * rng.beta(2, 2, n_regions) + 0.15 * (eco == "Northern") + 0.08 * (eco == "Prairie"), 0, 1
    )
    monitoring_density = np.clip(0.15 + 0.85 * urban_index + rng.normal(0, 0.12, n_regions), 0.02, 1.0)
    ecosystem_sensitivity = np.clip(
        0.30 + 0.35 * (eco == "Boreal") + 0.25 * (eco == "Northern") + 0.15 * rng.random(n_regions), 0, 1
    )

    out = pd.DataFrame(
        {
            "region": [f"R{i+1:02d}" for i in range(n_regions)],
            "ecozone": eco,
            "x": x,
            "y": y,
            "urban_index": urban_index,
            "road_density": road_density,
            "green_space": green_space,
            "population": pop,
            "vulnerability": vulnerability,
            "monitoring_density": monitoring_density,
            "ecosystem_sensitivity": ecosystem_sensitivity,
        }
    )
    return out


def make_time_index(start: str, end: str, freq: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq=freq)
    tdf = pd.DataFrame({"date": dates})
    tdf["year"] = tdf["date"].dt.year
    tdf["week"] = tdf["date"].dt.isocalendar().week.astype(int)
    tdf["month"] = tdf["date"].dt.month
    tdf["season_angle"] = 2 * np.pi * (tdf["week"] - 1) / 52.18
    tdf["summer"] = ((tdf["month"] >= 5) & (tdf["month"] <= 9)).astype(int)
    tdf["trend"] = (tdf["year"] - tdf["year"].min()) / (tdf["year"].max() - tdf["year"].min())
    return tdf


regions = make_regions(CFG.n_regions, rng)
time = make_time_index(CFG.start, CFG.end, CFG.freq)

# Cartesian product region x week.
df = regions.merge(time, how="cross")

# Regional baseline climate structure.
eco_temp_shift = {
    "Pacific": 2.0,
    "Prairie": -1.0,
    "Boreal": -3.0,
    "Atlantic": 0.5,
    "Northern": -8.0,
    "UrbanCorridor": 2.5,
}
eco_dry_shift = {
    "Pacific": -0.15,
    "Prairie": 0.20,
    "Boreal": 0.02,
    "Atlantic": -0.10,
    "Northern": 0.00,
    "UrbanCorridor": 0.06,
}

# Climate-change warming trend and weekly weather variability.
df["base_temp"] = df["ecozone"].map(eco_temp_shift).astype(float)
df["temperature"] = (
    8
    + df["base_temp"]
    + 13 * np.sin(df["season_angle"] - np.pi / 2)
    + 2.1 * df["trend"]
    + 1.2 * df["urban_index"]
    + rng.normal(0, 2.2, len(df))
)

df["heat_index"] = np.maximum(df["temperature"] - 24, 0)

df["drought"] = np.clip(
    0.30
    + df["ecozone"].map(eco_dry_shift).astype(float)
    + 0.28 * df["summer"]
    + 0.22 * df["trend"]
    + rng.normal(0, 0.13, len(df)),
    0,
    1,
)

df["wind_speed"] = np.clip(
    4.5
    + 1.1 * np.cos(df["season_angle"])
    - 0.9 * df["stagnation"] if "stagnation" in df.columns else 0,
    0,
    100,
)
# Define stagnation before final wind speed; lower wind and summer heat increase stagnation.
df["stagnation"] = np.clip(
    0.20 + 0.25 * df["summer"] + 0.18 * df["heat_index"] / 10 - 0.06 * rng.normal(0, 1, len(df)),
    0,
    1,
)
df["wind_speed"] = np.clip(5.8 - 2.8 * df["stagnation"] + rng.normal(0, 1.0, len(df)), 0.5, 10.0)
df["precipitation"] = np.clip(
    22 + 10 * np.cos(df["season_angle"]) - 18 * df["drought"] + rng.normal(0, 5, len(df)), 0, 70
)
df["boundary_layer_height"] = np.clip(
    450 + 35 * df["temperature"] - 250 * df["stagnation"] + rng.normal(0, 80, len(df)), 150, 2100
)
df["solar_radiation"] = np.clip(
    110 + 120 * df["summer"] + 7 * df["temperature"] + rng.normal(0, 25, len(df)), 30, 380
)

# Fire weather and urban/traffic emissions.
df["fire_weather"] = np.clip(
    0.15 + 0.35 * df["drought"] + 0.025 * df["temperature"] + 0.10 * df["summer"] - 0.005 * df["precipitation"], 0, 1
)
df["traffic_activity"] = np.clip(
    0.35 + 0.75 * df["urban_index"] + 0.18 * df["road_density"] + rng.normal(0, 0.05, len(df)), 0.05, 2.2
)

# Policy/transition variables: tailpipe declines after 2020; non-exhaust persists.
df["ev_share"] = np.clip(0.03 + 0.14 * np.maximum(df["year"] - 2020, 0) / 4 + 0.05 * df["urban_index"], 0, 0.45)
df["tailpipe_factor"] = 1 - 0.55 * df["ev_share"]
df["non_exhaust_factor"] = 0.25 + 0.75 * df["traffic_activity"]


# -----------------------------------------------------------------------------
# 2. Wildfire-smoke transport process
# -----------------------------------------------------------------------------

def simulate_fires(time: pd.DataFrame, regions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create weekly fire source points, mostly in summer and warmer years."""
    fire_records = []
    for _, row in time.iterrows():
        summer = row["summer"]
        trend = row["trend"]
        # More fire events in summer and later years.
        lam = 0.3 + 3.2 * summer + 2.2 * summer * trend
        n_fire = rng.poisson(lam)
        for j in range(n_fire):
            # Sources are more likely around boreal/northern zones but random in space.
            if rng.random() < 0.55:
                candidate = regions[regions["ecozone"].isin(["Boreal", "Northern", "Prairie"])]
                center = candidate.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
                fx = center["x"] + rng.normal(0, 120)
                fy = center["y"] + rng.normal(0, 120)
            else:
                fx = rng.uniform(0, 1000)
                fy = rng.uniform(0, 700)
            intensity = rng.gamma(shape=2.2, scale=18) * (1 + 1.1 * trend) * (1 + 0.65 * summer)
            fire_records.append(
                {
                    "date": row["date"],
                    "year": row["year"],
                    "week": row["week"],
                    "fire_id": f"F{row['year']}_{row['week']}_{j}",
                    "fx": fx,
                    "fy": fy,
                    "fire_intensity": intensity,
                }
            )
    return pd.DataFrame(fire_records)


fires = simulate_fires(time, regions, rng)

# Add smoke transport to each region-week.
df["smoke_true"] = 0.0
if len(fires) > 0:
    region_xy = regions[["x", "y"]].to_numpy()
    for date, fires_t in fires.groupby("date"):
        dist = cdist(region_xy, fires_t[["fx", "fy"]].to_numpy())
        # A distance-decay kernel with long tails: transported smoke can affect cities far from fires.
        kernel = np.exp(-dist / 220.0) / (1 + dist / 420.0)
        smoke_loading = kernel @ fires_t["fire_intensity"].to_numpy()
        idx = df["date"].eq(date)
        # Meteorology modifies observed smoke: stagnation increases, precipitation and wind remove.
        met = (
            1.0
            + 0.75 * df.loc[idx, "stagnation"].to_numpy()
            - 0.010 * df.loc[idx, "precipitation"].to_numpy()
            - 0.035 * df.loc[idx, "wind_speed"].to_numpy()
        )
        df.loc[idx, "smoke_true"] = np.maximum(0, smoke_loading * met)

# Scaled smoke influence for models.
df["smoke_scaled"] = df["smoke_true"] / (df["smoke_true"].quantile(0.95) + 1e-6)
df["smoke_scaled"] = np.clip(df["smoke_scaled"], 0, 3)


# -----------------------------------------------------------------------------
# 3. True air-quality process and imperfect mechanistic model
# -----------------------------------------------------------------------------

# True local emissions.
df["NOx_emission_true"] = 7.0 * df["traffic_activity"] * df["tailpipe_factor"] + 1.0 * df["urban_index"]
df["VOC_proxy_true"] = 1.2 + 0.9 * df["green_space"] * np.exp(0.025 * np.maximum(df["temperature"], 0)) + 0.45 * df["smoke_scaled"]
df["dust_nonexhaust_true"] = 1.3 * df["road_density"] * df["non_exhaust_factor"] * (1 + 0.45 * df["drought"])

# PM2.5 true process.
df["pm25_true_mean"] = (
    3.5
    + 4.8 * df["smoke_scaled"]
    + 1.1 * df["dust_nonexhaust_true"]
    + 0.32 * df["NOx_emission_true"]
    + 2.4 * df["stagnation"]
    - 0.035 * df["precipitation"]
    + 0.9 * df["drought"] * df["summer"]
)

# Ozone true process with nonlinear NOx-VOC chemistry and smoke/heat interaction.
voc_nox_ratio = df["VOC_proxy_true"] / (df["NOx_emission_true"] + 0.5)
df["ozone_regime_true"] = np.where(voc_nox_ratio > 0.55, "NOx-limited", "VOC-limited")
df["o3_true_mean"] = (
    20
    + 0.72 * np.maximum(df["temperature"], 0)
    + 0.035 * df["solar_radiation"]
    + 9.5 * np.tanh(1.8 * voc_nox_ratio)
    + 3.8 * df["stagnation"]
    + 4.0 * df["smoke_scaled"] * df["heat_index"] / 8.0
    - 1.1 * df["NOx_emission_true"] * (df["ozone_regime_true"].eq("VOC-limited"))
)

# NO2 true process.
df["no2_true_mean"] = (
    2.5
    + 4.2 * df["NOx_emission_true"]
    + 1.8 * df["stagnation"]
    - 0.35 * df["wind_speed"]
    - 0.45 * df["ev_share"] * df["traffic_activity"]
)

# Observed values with monitoring error. Under-monitored areas are noisier.
obs_noise_scale = 1.0 + 1.2 * (1 - df["monitoring_density"])
df["pm25_obs"] = np.maximum(0.5, df["pm25_true_mean"] + rng.normal(0, 1.1 * obs_noise_scale, len(df)))
df["o3_obs"] = np.maximum(1, df["o3_true_mean"] + rng.normal(0, 3.2 * obs_noise_scale, len(df)))
df["no2_obs"] = np.maximum(0.2, df["no2_true_mean"] + rng.normal(0, 2.1 * obs_noise_scale, len(df)))

# Imperfect mechanistic reduced-form model: intentionally misses some nonlinearities.
df["pm25_mech"] = (
    4.2
    + 3.9 * df["smoke_scaled"]
    + 1.0 * df["dust_nonexhaust_true"]
    + 0.24 * df["NOx_emission_true"]
    + 1.7 * df["stagnation"]
    - 0.025 * df["precipitation"]
)
df["o3_mech"] = (
    21
    + 0.60 * np.maximum(df["temperature"], 0)
    + 0.026 * df["solar_radiation"]
    + 6.0 * np.tanh(1.2 * voc_nox_ratio)
    + 2.1 * df["stagnation"]
)
df["no2_mech"] = 3.0 + 3.7 * df["NOx_emission_true"] + 1.0 * df["stagnation"] - 0.25 * df["wind_speed"]

# Risk indicators.
df["pm25_exceedance"] = (df["pm25_obs"] > 25).astype(int)
df["o3_exceedance"] = (df["o3_obs"] > 65).astype(int)
df["smoke_event"] = (df["smoke_scaled"] > df["smoke_scaled"].quantile(0.92)).astype(int)
df["heat_ozone_event"] = ((df["heat_index"] > 3) & (df["o3_obs"] > df["o3_obs"].quantile(0.85))).astype(int)

# Smoke-attributable fraction approximation, using true smoke component.
df["smoke_attrib_fraction_true"] = np.clip((4.8 * df["smoke_scaled"]) / (df["pm25_true_mean"] + 1e-6), 0, 1)
df["deposition_risk"] = np.clip(
    0.02 * df["pm25_obs"] + 0.012 * df["no2_obs"] + 0.45 * df["ecosystem_sensitivity"] + 0.20 * df["precipitation"] / 70,
    0,
    2,
)


# -----------------------------------------------------------------------------
# 4. Train/test split and modelling functions
# -----------------------------------------------------------------------------

FEATURES = [
    "temperature",
    "heat_index",
    "drought",
    "stagnation",
    "wind_speed",
    "precipitation",
    "boundary_layer_height",
    "solar_radiation",
    "fire_weather",
    "smoke_scaled",
    "urban_index",
    "road_density",
    "green_space",
    "traffic_activity",
    "ev_share",
    "tailpipe_factor",
    "non_exhaust_factor",
    "NOx_emission_true",
    "VOC_proxy_true",
    "dust_nonexhaust_true",
    "vulnerability",
    "monitoring_density",
    "ecosystem_sensitivity",
    "week",
    "trend",
]

TARGETS = {
    "PM2.5": {"obs": "pm25_obs", "mech": "pm25_mech", "threshold": 25},
    "Ozone": {"obs": "o3_obs", "mech": "o3_mech", "threshold": 65},
    "NO2": {"obs": "no2_obs", "mech": "no2_mech", "threshold": 35},
}

train = df[df["year"] <= 2022].copy()
test = df[df["year"] >= 2023].copy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}


def fit_models_for_target(target_name: str) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    spec = TARGETS[target_name]
    y_train = train[spec["obs"]].values
    y_test = test[spec["obs"]].values
    X_train = train[FEATURES]
    X_test = test[FEATURES]

    # Mechanistic-only prediction.
    pred_mech = test[spec["mech"]].values

    # Pure AI predicts observed concentration directly.
    pure_ai = RandomForestRegressor(
        n_estimators=60, min_samples_leaf=8, random_state=CFG.seed, n_jobs=1
    )
    pure_ai.fit(X_train, y_train)
    pred_ai = pure_ai.predict(X_test)

    # Hybrid AI predicts residual = observed - mechanistic.
    resid_train = train[spec["obs"]] - train[spec["mech"]]
    resid_model = GradientBoostingRegressor(
        n_estimators=90,
        learning_rate=0.06,
        max_depth=3,
        subsample=0.75,
        random_state=CFG.seed,
    )
    resid_model.fit(X_train, resid_train)
    pred_hybrid = test[spec["mech"]].values + resid_model.predict(X_test)

    rows = []
    for model_name, pred in [
        ("Mechanistic only", pred_mech),
        ("Pure AI", pred_ai),
        ("Hybrid mechanistic + AI residual", pred_hybrid),
    ]:
        metrics = evaluate(y_test, pred)
        metrics.update({"Target": target_name, "Model": model_name, "Subset": "All test weeks 2023-2024"})
        rows.append(metrics)

        # Extreme-event subset.
        if target_name == "PM2.5":
            mask = test["smoke_event"].values.astype(bool)
            subset = "Smoke-event weeks"
        elif target_name == "Ozone":
            mask = test["heat_ozone_event"].values.astype(bool)
            subset = "Heat/ozone-event weeks"
        else:
            mask = test["traffic_activity"].values > test["traffic_activity"].quantile(0.75)
            subset = "High-traffic weeks"
        if mask.sum() > 10:
            metrics_e = evaluate(y_test[mask], pred[mask])
            metrics_e.update({"Target": target_name, "Model": model_name, "Subset": subset})
            rows.append(metrics_e)

    # Prediction frame for plots.
    pred_df = test[["date", "year", "week", "region", "ecozone", spec["obs"], spec["mech"], "smoke_event", "heat_ozone_event"]].copy()
    pred_df = pred_df.rename(columns={spec["obs"]: "observed", spec["mech"]: "mechanistic"})
    pred_df["pure_ai"] = pred_ai
    pred_df["hybrid"] = pred_hybrid
    pred_df["target"] = target_name

    models = {"pure_ai": pure_ai, "resid_model": resid_model}
    return pd.DataFrame(rows), models, pred_df


all_metrics = []
models_by_target = {}
predictions = []
for target in TARGETS:
    metrics_t, models_t, pred_t = fit_models_for_target(target)
    all_metrics.append(metrics_t)
    models_by_target[target] = models_t
    predictions.append(pred_t)

metrics = pd.concat(all_metrics, ignore_index=True)
predictions = pd.concat(predictions, ignore_index=True)

metrics_path = os.path.join(CFG.output_dir, "model_performance_metrics.csv")
pred_path = os.path.join(CFG.output_dir, "test_predictions_2022_2024.csv")
df_path = os.path.join(CFG.output_dir, "synthetic_climair_dataset.csv")
metrics.to_csv(metrics_path, index=False)
predictions.to_csv(pred_path, index=False)
df.to_csv(df_path, index=False)


# -----------------------------------------------------------------------------
# 5. Bootstrap uncertainty for PM2.5 hybrid model
# -----------------------------------------------------------------------------

def bootstrap_hybrid_uncertainty(target_name: str = "PM2.5") -> pd.DataFrame:
    spec = TARGETS[target_name]
    X_train = train[FEATURES].reset_index(drop=True)
    X_test = test[FEATURES].reset_index(drop=True)
    resid_train = (train[spec["obs"]] - train[spec["mech"]]).reset_index(drop=True)
    pred_boot = []
    n = len(X_train)
    for b in range(CFG.n_bootstrap_models):
        idx = rng.integers(0, n, n)
        model = GradientBoostingRegressor(
            n_estimators=60,
            learning_rate=0.07,
            max_depth=3,
            subsample=0.75,
            random_state=1000 + b,
        )
        model.fit(X_train.iloc[idx], resid_train.iloc[idx])
        pred = test[spec["mech"]].values + model.predict(X_test)
        pred_boot.append(pred)
    boot = np.vstack(pred_boot)
    out = test[["date", "region", "ecozone", "pm25_obs", "pm25_mech", "smoke_event", "vulnerability", "monitoring_density"]].copy()
    out["hybrid_mean"] = boot.mean(axis=0)
    out["hybrid_p05"] = np.percentile(boot, 5, axis=0)
    out["hybrid_p95"] = np.percentile(boot, 95, axis=0)
    out["interval_width"] = out["hybrid_p95"] - out["hybrid_p05"]
    out["covered"] = ((out["pm25_obs"] >= out["hybrid_p05"]) & (out["pm25_obs"] <= out["hybrid_p95"])).astype(int)
    return out


uncertainty_pm25 = bootstrap_hybrid_uncertainty("PM2.5")
unc_path = os.path.join(CFG.output_dir, "pm25_hybrid_uncertainty_bootstrap.csv")
uncertainty_pm25.to_csv(unc_path, index=False)


# -----------------------------------------------------------------------------
# 6. Future scenario engine for 2035 and 2050
# -----------------------------------------------------------------------------

def make_future_scenario(base: pd.DataFrame, scenario_year: int, scenario_name: str) -> pd.DataFrame:
    """Create future forcing by modifying a recent baseline year."""
    recent = base[base["year"] == 2024].copy()
    recent["year"] = scenario_year
    recent["date"] = recent["date"] + pd.DateOffset(years=scenario_year - 2024)
    delta_years = scenario_year - 2024

    # Scenario assumptions.
    if scenario_name == "moderate_transition":
        warming = 0.035 * delta_years
        wildfire_multiplier = 1.0 + 0.025 * delta_years
        ev_add = 0.018 * delta_years
        stagnation_add = 0.004 * delta_years
    elif scenario_name == "high_wildfire_heat":
        warming = 0.055 * delta_years
        wildfire_multiplier = 1.0 + 0.055 * delta_years
        ev_add = 0.010 * delta_years
        stagnation_add = 0.007 * delta_years
    elif scenario_name == "fast_transport_electrification":
        warming = 0.035 * delta_years
        wildfire_multiplier = 1.0 + 0.025 * delta_years
        ev_add = 0.035 * delta_years
        stagnation_add = 0.004 * delta_years
    else:
        raise ValueError("Unknown scenario")

    recent["scenario"] = scenario_name
    recent["temperature"] += warming
    recent["heat_index"] = np.maximum(recent["temperature"] - 24, 0)
    recent["drought"] = np.clip(recent["drought"] + 0.006 * delta_years, 0, 1)
    recent["stagnation"] = np.clip(recent["stagnation"] + stagnation_add, 0, 1)
    recent["precipitation"] = np.clip(recent["precipitation"] - 0.12 * delta_years * recent["summer"], 0, 75)
    recent["fire_weather"] = np.clip(
        0.15 + 0.35 * recent["drought"] + 0.025 * recent["temperature"] + 0.10 * recent["summer"] - 0.005 * recent["precipitation"],
        0,
        1,
    )
    recent["smoke_scaled"] = np.clip(recent["smoke_scaled"] * wildfire_multiplier, 0, 4)
    recent["ev_share"] = np.clip(recent["ev_share"] + ev_add, 0, 0.95)
    recent["tailpipe_factor"] = 1 - 0.55 * recent["ev_share"]
    recent["non_exhaust_factor"] = 0.25 + 0.75 * recent["traffic_activity"]

    recent["NOx_emission_true"] = 7.0 * recent["traffic_activity"] * recent["tailpipe_factor"] + 1.0 * recent["urban_index"]
    recent["VOC_proxy_true"] = 1.2 + 0.9 * recent["green_space"] * np.exp(0.025 * np.maximum(recent["temperature"], 0)) + 0.45 * recent["smoke_scaled"]
    recent["dust_nonexhaust_true"] = 1.3 * recent["road_density"] * recent["non_exhaust_factor"] * (1 + 0.45 * recent["drought"])

    voc_nox = recent["VOC_proxy_true"] / (recent["NOx_emission_true"] + 0.5)
    recent["pm25_mech"] = (
        4.2
        + 3.9 * recent["smoke_scaled"]
        + 1.0 * recent["dust_nonexhaust_true"]
        + 0.24 * recent["NOx_emission_true"]
        + 1.7 * recent["stagnation"]
        - 0.025 * recent["precipitation"]
    )
    recent["o3_mech"] = (
        21
        + 0.60 * np.maximum(recent["temperature"], 0)
        + 0.026 * recent["solar_radiation"]
        + 6.0 * np.tanh(1.2 * voc_nox)
        + 2.1 * recent["stagnation"]
    )
    recent["no2_mech"] = 3.0 + 3.7 * recent["NOx_emission_true"] + 1.0 * recent["stagnation"] - 0.25 * recent["wind_speed"]
    return recent


scenario_frames = []
for yr in [2035, 2050]:
    for scen in ["moderate_transition", "high_wildfire_heat", "fast_transport_electrification"]:
        scenario_frames.append(make_future_scenario(df, yr, scen))
future = pd.concat(scenario_frames, ignore_index=True)

# Apply trained hybrid residual models to future scenario forcing.
for target, spec in TARGETS.items():
    model = models_by_target[target]["resid_model"]
    future[f"{target}_hybrid"] = future[spec["mech"]] + model.predict(future[FEATURES])

future["smoke_attrib_fraction_est"] = np.clip((4.8 * future["smoke_scaled"]) / (future["PM2.5_hybrid"] + 1e-6), 0, 1)
future["pm25_exceedance_prob_proxy"] = 1 / (1 + np.exp(-(future["PM2.5_hybrid"] - 25) / 4))
future["o3_exceedance_prob_proxy"] = 1 / (1 + np.exp(-(future["Ozone_hybrid"] - 65) / 5))
future["deposition_risk_future"] = np.clip(
    0.02 * future["PM2.5_hybrid"] + 0.012 * future["NO2_hybrid"] + 0.45 * future["ecosystem_sensitivity"] + 0.20 * future["precipitation"] / 70,
    0,
    2,
)
future["health_vulnerability_risk"] = future["vulnerability"] * (0.65 * future["pm25_exceedance_prob_proxy"] + 0.35 * future["o3_exceedance_prob_proxy"])
future["monitoring_gap_score"] = np.clip(1 - future["monitoring_density"] + 0.25 * future["pm25_exceedance_prob_proxy"], 0, 1.25)

# Pollutant-priority ranking by region-scenario-year.
priority = (
    future.groupby(["scenario", "year", "region", "ecozone"])
    .agg(
        PM25_mean=("PM2.5_hybrid", "mean"),
        Ozone_mean=("Ozone_hybrid", "mean"),
        NO2_mean=("NO2_hybrid", "mean"),
        Smoke_fraction=("smoke_attrib_fraction_est", "mean"),
        PM25_exceed_prob=("pm25_exceedance_prob_proxy", "mean"),
        Ozone_exceed_prob=("o3_exceedance_prob_proxy", "mean"),
        Deposition_risk=("deposition_risk_future", "mean"),
        Health_vulnerability_risk=("health_vulnerability_risk", "mean"),
        Monitoring_gap_score=("monitoring_gap_score", "mean"),
    )
    .reset_index()
)
priority["Future_pollutant_priority_score"] = (
    0.27 * priority["PM25_exceed_prob"]
    + 0.22 * priority["Ozone_exceed_prob"]
    + 0.16 * priority["Smoke_fraction"]
    + 0.16 * priority["Deposition_risk"] / priority["Deposition_risk"].max()
    + 0.12 * priority["Health_vulnerability_risk"]
    + 0.07 * priority["Monitoring_gap_score"] / priority["Monitoring_gap_score"].max()
)
priority["priority_rank_within_scenario"] = priority.groupby(["scenario", "year"])["Future_pollutant_priority_score"].rank(ascending=False)

future_path = os.path.join(CFG.output_dir, "future_scenarios_2035_2050.csv")
priority_path = os.path.join(CFG.output_dir, "future_pollutant_priority_scores.csv")
future.to_csv(future_path, index=False)
priority.to_csv(priority_path, index=False)


# -----------------------------------------------------------------------------
# 7. Figures
# -----------------------------------------------------------------------------

plt.rcParams.update({"font.size": 10})


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# Figure 1: PM2.5 observed vs predictions.
pm_pred = predictions[predictions["target"] == "PM2.5"].copy()
plt.figure(figsize=(7.2, 5.5))
plt.scatter(pm_pred["observed"], pm_pred["mechanistic"], s=14, alpha=0.35, label="Mechanistic only")
plt.scatter(pm_pred["observed"], pm_pred["pure_ai"], s=14, alpha=0.35, label="Pure AI")
plt.scatter(pm_pred["observed"], pm_pred["hybrid"], s=14, alpha=0.35, label="Hybrid")
lims = [0, max(pm_pred[["observed", "mechanistic", "pure_ai", "hybrid"]].max()) * 1.05]
plt.plot(lims, lims, linestyle="--", linewidth=1)
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Observed PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("PM2.5 prediction: mechanistic, pure AI, and hybrid models")
plt.legend(frameon=False)
save_fig(os.path.join(CFG.output_dir, "fig1_pm25_observed_vs_predicted.png"))

# Figure 2: Model performance by target and model.
metrics_all = metrics[metrics["Subset"].str.contains("All test")].copy()
order = ["Mechanistic only", "Pure AI", "Hybrid mechanistic + AI residual"]
for metric in ["RMSE", "MAE", "R2"]:
    pivot = metrics_all.pivot(index="Target", columns="Model", values=metric)[order]
    ax = pivot.plot(kind="bar", figsize=(7.4, 4.8))
    ax.set_ylabel(metric)
    ax.set_title(f"Blocked validation performance, 2023-2024: {metric}")
    ax.legend(frameon=False, loc="best")
    plt.xticks(rotation=0)
    save_fig(os.path.join(CFG.output_dir, f"fig2_model_performance_{metric}.png"))

# Figure 3: Smoke attribution map-like scatter for future priority.
top_scen = priority[(priority["year"] == 2050) & (priority["scenario"] == "high_wildfire_heat")].merge(
    regions[["region", "x", "y", "population"]], on="region", how="left"
)
plt.figure(figsize=(7.6, 5.4))
sc = plt.scatter(
    top_scen["x"],
    top_scen["y"],
    s=30 + 150 * top_scen["population"] / top_scen["population"].max(),
    c=top_scen["Future_pollutant_priority_score"],
    alpha=0.80,
)
plt.colorbar(sc, label="Pollutant-priority score")
plt.xlabel("Synthetic west-east coordinate")
plt.ylabel("Synthetic south-north coordinate")
plt.title("2050 high-wildfire/heat scenario: regional priority score")
save_fig(os.path.join(CFG.output_dir, "fig3_2050_priority_map_like_scatter.png"))

# Figure 4: Uncertainty intervals for a high-risk region.
high_region = (
    uncertainty_pm25.groupby("region")["interval_width"].mean().sort_values(ascending=False).index[0]
)
u = uncertainty_pm25[uncertainty_pm25["region"] == high_region].sort_values("date")
plt.figure(figsize=(8.0, 4.8))
plt.plot(u["date"], u["pm25_obs"], linewidth=1.3, label="Observed")
plt.plot(u["date"], u["hybrid_mean"], linewidth=1.3, label="Hybrid mean")
plt.fill_between(u["date"], u["hybrid_p05"], u["hybrid_p95"], alpha=0.25, label="Bootstrap 90% interval")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title(f"Hybrid PM2.5 uncertainty interval: {high_region}")
plt.legend(frameon=False)
save_fig(os.path.join(CFG.output_dir, "fig4_pm25_bootstrap_uncertainty.png"))

# Figure 5: Scenario summary by year and scenario.
scenario_summary = (
    priority.groupby(["scenario", "year"])
    .agg(
        mean_priority=("Future_pollutant_priority_score", "mean"),
        mean_pm25_exceed_prob=("PM25_exceed_prob", "mean"),
        mean_o3_exceed_prob=("Ozone_exceed_prob", "mean"),
        mean_smoke_fraction=("Smoke_fraction", "mean"),
        mean_monitoring_gap=("Monitoring_gap_score", "mean"),
    )
    .reset_index()
)
scenario_summary.to_csv(os.path.join(CFG.output_dir, "scenario_summary.csv"), index=False)

for col, ylabel in [
    ("mean_pm25_exceed_prob", "Mean PM2.5 exceedance probability proxy"),
    ("mean_o3_exceed_prob", "Mean ozone exceedance probability proxy"),
    ("mean_smoke_fraction", "Mean smoke-attributable fraction"),
]:
    plt.figure(figsize=(7.2, 4.8))
    for scen, g in scenario_summary.groupby("scenario"):
        plt.plot(g["year"], g[col], marker="o", linewidth=1.8, label=scen)
    plt.xlabel("Scenario year")
    plt.ylabel(ylabel)
    plt.title(f"Future scenario comparison: {ylabel}")
    plt.legend(frameon=False)
    save_fig(os.path.join(CFG.output_dir, f"fig5_scenario_{col}.png"))

# Figure 6: Feature importance for hybrid residual correction.
for target in ["PM2.5", "Ozone"]:
    model = models_by_target[target]["resid_model"]
    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=False).head(12)
    plt.figure(figsize=(7.2, 5.0))
    plt.barh(imp["feature"][::-1], imp["importance"][::-1])
    plt.xlabel("Gradient boosting importance for residual correction")
    plt.title(f"Hybrid AI residual drivers: {target}")
    save_fig(os.path.join(CFG.output_dir, f"fig6_feature_importance_{target.replace('.', '')}.png"))


# -----------------------------------------------------------------------------
# 8. Console report
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("CLIMAIR-CANADA PROOF-OF-CONCEPT SIMULATION COMPLETE")
print("=" * 78)
print(f"Synthetic dataset rows: {len(df):,}")
print(f"Regions: {regions.shape[0]}, weekly dates: {time.shape[0]}, simulated fires: {len(fires):,}")
print("\nBlocked validation metrics, all test weeks 2023-2024:")
print(metrics_all[["Target", "Model", "RMSE", "MAE", "R2", "Bias"]].round(3).to_string(index=False))

print("\nExtreme-event metrics:")
print(metrics[~metrics["Subset"].str.contains("All test")][["Target", "Subset", "Model", "RMSE", "MAE", "R2", "Bias"]].round(3).to_string(index=False))

coverage = uncertainty_pm25["covered"].mean()
width = uncertainty_pm25["interval_width"].mean()
print(f"\nPM2.5 bootstrap uncertainty: empirical coverage={coverage:.3f}, mean interval width={width:.2f}")

print("\nFuture scenario summary:")
print(scenario_summary.round(3).to_string(index=False))

print("\nTop 10 region-scenario priority scores:")
print(
    priority.sort_values("Future_pollutant_priority_score", ascending=False)
    .head(10)[["scenario", "year", "region", "ecozone", "Future_pollutant_priority_score", "PM25_exceed_prob", "Ozone_exceed_prob", "Smoke_fraction", "Monitoring_gap_score"]]
    .round(3)
    .to_string(index=False)
)

print("\nFiles saved in:", os.path.abspath(CFG.output_dir))
for fn in sorted(os.listdir(CFG.output_dir)):
    print(" -", fn)
