# team_flowcode — Addis Ababa ride-demand forecast

**Team:** flowcode · **Members:** Amina Aman, Niyat Debesay, Henok Solomon, Zakria Abdurahman · **Competition:** Flowcode Addis Ababa Ride-Demand Forecast, 2025 · **Date:** 2024-06-19

## Summary

We forecast hourly ride demand (trips per zone-hour) for 12 Addis Ababa zones for 1–14 November 2025. Three messy exports — trips, hourly weather and an events calendar — are cleaned in code (55 zone spellings → 12, three timestamp formats and two clocks → one Addis-time clock, ×8 trip spikes, °F temperatures, placeholder codes, duplicates), joined with trips as the left table, and turned into 31 features that are all known 14 days ahead (calendar, zone, trend and 14–28-day lags, weather, events). A tuned LightGBM model, chosen on four rolling 14-day validation windows, beats the seasonal-naive baseline by 14% and is served in a Streamlit app that looks up weather and events for any zone and day and returns the hourly forecast, drivers needed and expected fares.

| Final validation score (4 rolling 14-day folds, Sep–Oct 2025) | |
|---|---|
| RMSE | **9.39 ± 0.86** trips per zone-hour |
| MAE | **6.07** trips per zone-hour (≈ 18% of mean demand, ≈ 5 drivers) |
| Seasonal-naive baseline RMSE | 10.98 |

**Demo:** _add the Streamlit Community Cloud URL here after deploying_ — or run locally (below).

## Setup

Python 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

Run the notebooks top to bottom, in this order (each reads only `data/raw/` and the outputs of the ones before it):

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_cleaning_and_integration.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_analysis_report.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_visualizations.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/04_modeling_and_evaluation.ipynb
```

Notebook 04 takes about 15 minutes (30-trial tuning and four rolling folds). Then export the reports:

```bash
python -m nbconvert --to html --no-input notebooks/01_cleaning_and_integration.ipynb --output-dir reports --output A_cleaning_and_integration
python -m nbconvert --to html --no-input notebooks/02_analysis_report.ipynb --output-dir reports --output B_analysis_report
python -m nbconvert --to html --no-input notebooks/04_modeling_and_evaluation.ipynb --output-dir reports --output D_model_evaluation
```

Run the demo app:

```bash
streamlit run app/app.py
```

## Where each deliverable lives

| Deliverable | Location |
|---|---|
| Prediction | `submission/team_flowcode_submission.csv` (`row_id, predicted_trips`, 4,032 rows, original order) |
| A — Cleaning & integration | `notebooks/01_cleaning_and_integration.ipynb`, `reports/A_cleaning_and_integration.html`, `data/processed/master_train.csv`, `master_test.csv`, `data_dictionary_master.csv`, `cleaning_log.csv`, `feature_table.csv` |
| B — Analysis report | `notebooks/02_analysis_report.ipynb`, `reports/B_analysis_report.html` |
| C — Visualizations | `figures/fig01_…png` – `fig12_…png`, `figures/figure_captions.md`, `notebooks/03_visualizations.ipynb` |
| D — Modeling & evaluation | `notebooks/04_modeling_and_evaluation.ipynb`, `reports/D_model_evaluation.html`, `models/final_model.joblib` |
| E — Demo app | `app/app.py`, `app/assets/` (bundled model, November features, weather, events, fares, typical profiles), `app/requirements.txt` |
| F — Slides | `presentation/team_flowcode_slides.pdf` |
| Pipeline explained + assumptions | `reports/pipeline_explained.md` |

## Folder layout

```
README.md  requirements.txt  .gitignore
submission/team_flowcode_submission.csv
data/raw/                 original CSVs, never edited
data/processed/           cleaned tables, master_train.csv, master_test.csv, data_dictionary_master.csv, logs
notebooks/01_… – 04_…     run in order
models/final_model.joblib (+ LightGBM text copy and metadata)
figures/                  fig01–fig12, figure_captions.md, supporting figures
reports/                  A_, B_, D_ HTML reports, pipeline_explained.md
app/app.py, app/assets/, app/requirements.txt
presentation/
```

## Method hygiene

- **Rule 5:** the final model uses weather (temperature, rain, rain over 3 h, rain class, humidity, wind) and event features (in-window flags, attendance, hours to/since an event, type flags).
- **Rule 6:** only forecast-time features — demand lags are at least 14 days old; same-hour `active_drivers`, `avg_wait_min` and `avg_fare_birr` are excluded (leakage demo in D4).
- **Rule 7:** validation is chronological — rolling origin, cut dates 6 Sep, 20 Sep, 4 Oct, 18 Oct, each validating the next 14 days; tuning used an earlier split (23 Aug – 5 Sep).
- **Rule 8:** every statistic (imputation medians, thresholds, baselines) is learned from train only; the test file is cleaned and featurised with train statistics and never used for fitting.
- Fixed random seeds (`random_state=42` for models, `0` for clustering); relative paths only.
