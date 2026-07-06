# CLIMAIR-Canada Proof-of-Concept Python Simulation

## Overview

This repository-style proof of concept supports the proposal **CLIMAIR-Canada: Climate-Air Quality Intelligence for Wildfire, Urban, and Transportation Emission Risks**. The script demonstrates, in miniature, how an open hybrid mechanistic-AI platform could integrate climate, wildfire, urban, transportation, atmospheric chemistry, exposure, ecosystem, and uncertainty information into policy-relevant air-quality indicators.

The script is **not an operational forecast model** and does not use real monitoring data. Instead, it creates a scientifically structured synthetic dataset that mimics the logic of the proposed CLIMAIR-Canada system. This allows the full workflow to be demonstrated reproducibly without requiring access to federal, provincial, satellite, or proprietary transportation datasets.

## Main script

```text
climair_canada_poc.py
```

Running this script will generate a complete proof-of-concept analysis and save all outputs to:

```text
climair_poc_outputs/
```

## What the script demonstrates

The script demonstrates the following CLIMAIR-Canada capabilities:

1. Creation of a harmonized spatiotemporal climate-air-quality dataset.
2. Simulation of Canadian regional differences in urbanization, road density, forest exposure, population vulnerability, ecosystem sensitivity, and monitoring density.
3. Simulation of weekly climate and meteorological variables, including temperature, drought, wind, precipitation, stagnation, boundary-layer mixing, sunlight, and fire-weather conditions.
4. Simulation of wildfire-smoke influence using fire activity, drought, wind-informed transport, distance decay, and forest exposure.
5. Simulation of urban and transportation emissions, including NOx, VOC proxies, tailpipe effects, and non-exhaust particulate matter.
6. Mechanistic reduced-form prediction of PM2.5, ozone, and NO2.
7. A pure AI baseline model using machine learning directly on environmental and source features.
8. A hybrid mechanistic-AI residual model in which machine learning corrects the residuals left by the mechanistic model.
9. Temporal blocked validation, where earlier years are used for training and later years are held out for testing.
10. Extreme-event evaluation for wildfire-smoke weeks, heat/ozone weeks, and high-traffic weeks.
11. Bootstrap uncertainty intervals for hybrid PM2.5 prediction.
12. Scenario analysis for 2035 and 2050 under multiple future climate and emissions assumptions.
13. Pollutant-priority scoring, smoke-attribution estimation, deposition-risk indicators, health-vulnerability indicators, and monitoring-gap indicators.
14. Export of publication-style figures and reproducible CSV tables.

## Required Python packages

The script uses standard scientific Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

Recommended Python version:

```text
Python 3.9 or newer
```

The script has no internet requirement and should run locally after the required packages are installed.

## How to run

Place the script in a working folder and run:

```bash
python climair_canada_poc.py
```

In Jupyter Notebook or JupyterLab, you can run it from a notebook cell with:

```python
%run climair_canada_poc.py
```

The script will print a console summary and create an output folder called:

```text
climair_poc_outputs
```

## Expected runtime

On a typical laptop, the script should usually finish in under one minute. Runtime may vary depending on CPU speed and Python environment.

## Output files

The script saves the following core CSV files:

`synthetic_climair_dataset.csv` Full synthetic region-week dataset containing climate, emissions, pollutant, vulnerability, ecosystem, and monitoring variables.

`model_performance_metrics.csv` Performance metrics for mechanistic-only, pure AI, and hybrid models. Includes all-test and extreme-event subsets.

`test_predictions_2022_2024.csv` Observed and predicted pollutant values for the held-out test period.

`pm25_hybrid_uncertainty_bootstrap.csv` Bootstrap uncertainty intervals for hybrid PM2.5 predictions.

`future_scenarios_2035_2050.csv` Future scenario outputs for 2035 and 2050.

`future_pollutant_priority_scores.csv` Regional pollutant-priority scores by year and scenario.

`scenario_summary.csv` Aggregated scenario summaries comparing future risk indicators.

The script also saves the following figures:

`fig1_pm25_observed_vs_predicted.png` Observed-versus-predicted PM2.5 comparison for mechanistic-only, pure AI, and hybrid models.

`fig2_model_performance_RMSE.png` RMSE comparison across models and pollutants.

`fig2_model_performance_MAE.png` MAE comparison across models and pollutants.

`fig2_model_performance_R2.png` R² comparison across models and pollutants.

`fig3_2050_priority_map_like_scatter.png` Synthetic map-like regional pollutant-priority scores for the 2050 high-wildfire/heat scenario.

`fig4_pm25_bootstrap_uncertainty.png` PM2.5 hybrid prediction with bootstrap uncertainty interval for a high-risk region.

`fig5_scenario_mean_pm25_exceed_prob.png` Future scenario comparison for PM2.5 exceedance-probability proxy.

`fig5_scenario_mean_o3_exceed_prob.png` Future scenario comparison for ozone exceedance-probability proxy.

`fig5_scenario_mean_smoke_fraction.png` Future scenario comparison for smoke-attributable fraction.

`fig6_feature_importance_PM25.png` Feature importance for the PM2.5 hybrid residual-correction model.

`fig6_feature_importance_Ozone.png` Feature importance for the ozone hybrid residual-correction model.

## Model design

The script compares three modelling strategies.

### 1. Mechanistic-only model

The mechanistic model uses simplified process-based equations to represent how emissions and climate drivers affect pollutants. For example, PM2.5 is influenced by wildfire smoke, non-exhaust transportation particles, NOx-related urban emissions, stagnation, precipitation removal, and boundary-layer mixing. Ozone is influenced by temperature, sunlight, NOx-VOC balance, smoke interaction, and stagnation. NO2 is driven mainly by transportation and urban NOx emissions, with weather-driven dispersion.

This model is interpretable but deliberately imperfect, as real atmospheric systems include unresolved nonlinearities and local heterogeneity.

### 2. Pure AI model

The pure AI model uses machine learning directly on the environmental and emissions feature set to predict each pollutant. This provides a flexible predictive benchmark but is less process-explicit.

### 3. Hybrid mechanistic-AI residual model

The hybrid model first uses the mechanistic model to generate a baseline prediction. It then trains a machine-learning model on the residual error:

```text
Observed pollutant - Mechanistic prediction = residual structure
```

The final hybrid prediction is:

```text
Hybrid prediction = Mechanistic prediction + AI-predicted residual
```

This reflects the CLIMAIR-Canada proposal philosophy: artificial intelligence is used to improve prediction and capture unresolved nonlinear structure, but the process-based environmental model remains the scientific backbone.

## Validation design

The script uses temporal blocked validation. It trains models on earlier years and tests them on later held-out years. This is more realistic than random splitting because air-quality systems must generalize to future conditions, not merely interpolate among randomly shuffled observations.

The validation reports:

- R²
- RMSE
- MAE
- bias
- extreme-event performance

Extreme-event subsets include:

- wildfire-smoke weeks for PM2.5,
- heat/ozone weeks for ozone,
- high-traffic weeks for NO2.

## Future scenarios

The script simulates three scenario families for 2035 and 2050:

`moderate_transition` Warmer climate with moderate wildfire increase and moderate transportation transition.

`high_wildfire_heat` Stronger warming, stronger wildfire activity, and greater stagnation risk.

`fast_transport_electrification` Stronger electric-vehicle transition with remaining non-exhaust PM and climate-driven risks.


The scenario module produces future estimates of:

- PM2.5,
- ozone,
- NO2,
- smoke-attributable PM2.5 fraction,
- PM2.5 exceedance-probability proxy,
- ozone exceedance-probability proxy,
- deposition risk,
- vulnerability-weighted health risk,
- monitoring-gap score,
- future pollutant-priority score.

## How to interpret the outputs

The proof-of-concept results should be interpreted as a feasibility demonstration rather than empirical evidence about actual Canadian air quality. The synthetic data are designed to show that the proposed modelling architecture can be implemented end-to-end.

A successful run demonstrates that:

1. A harmonized climate-air-quality data structure can be created.
2. Wildfire, urban, transportation, and climate drivers can be represented together.
3. Reduced-form mechanistic equations can generate interpretable pollutant predictions.
4. Machine learning can be used as a residual-correction layer rather than an opaque replacement for process modelling.
5. Blocked validation can compare mechanistic-only, pure AI, and hybrid models.
6. Extreme-event performance can be evaluated separately from average conditions.
7. Future scenarios can be translated into policy-relevant risk indicators.
8. Uncertainty, monitoring gaps, vulnerability, ecosystem sensitivity, and pollutant priorities can be reported as decision-support outputs.

## Reproducibility

The script uses a fixed random seed defined in the configuration section:

```python
seed: int = 42
```

Changing the seed will generate a different synthetic dataset but should preserve the same general workflow and modelling logic.

## Configuration

At the top of the script, the `Config` dataclass controls the main simulation settings:

```python
@dataclass
class Config:
    seed: int = 42
    n_regions: int = 18
    start: str = "2018-01-01"
    end: str = "2024-12-31"
    freq: str = "W-MON"
    output_dir: str = "climair_poc_outputs"
    n_bootstrap_models: int = 6
```

You can modify these values to increase the number of synthetic regions, change the time window, rename the output directory, or increase the number of bootstrap models used for uncertainty estimation.

## Important limitations

This script intentionally uses synthetic data. Therefore:

- it should not be used to make real policy claims,
- it should not be interpreted as a Canadian air-quality forecast,
- it does not estimate real regional exposure,
- it does not replace ECCC, provincial, satellite, or operational forecast products,
- it is intended only to demonstrate scientific and computational feasibility.

The next development step would be to replace synthetic inputs with real harmonized data streams and validate model outputs against observed monitoring and satellite products.