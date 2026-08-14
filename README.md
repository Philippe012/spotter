# Spotter Freight Rate Prediction

Machine Learning Engineer Assessment submission — predicts freight spot rates
using a tuned LightGBM + XGBoost ensemble, validated on a chronological holdout.

## Setup

```bash
python -m venv venv
source venv/bin/activate     
pip install -r requirements.txt
```

Place the assessment data files in `data/raw/`:
- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december_chart_inputs.csv`

## Run

```bash
python run.py
```

This runs the full pipeline end-to-end:
load data → preprocess/clean → engineer features → chronological train/validation
split → tune & train the LightGBM+XGBoost ensemble (Optuna) → evaluate on the
holdout → generate predictions for `validation.csv` → generate the December
scenario predictions → validate everything and produce the chart via the
provided `score.py`.

**Useful flags:**
- `--skip-tune` — use default hyperparameters instead of running Optuna (faster)
- `--n-trials N` — set the number of Optuna trials (default 50)
- `--predict-only` — skip training and use the already-saved model

## Outputs

| File | Description |
|---|---|
| `outputs/predictions/validation_predictions.csv` | Final `load_id,predicted_rate` submission file |
| `outputs/predictions/december_predictions.csv` | December scenario predictions |
| `outputs/figures/residual_plot.png` | Residual diagnostics |
| `outputs/figures/feature_importance.png` | Top-20 feature importances |
| `outputs/reports/model_report.txt` | Full metrics report |
| `scorer_results/candidate_december.png` | Official December chart from `score.py` |

## Project structure

spotter/
├── data/raw/ — input CSVs (not committed; see Setup)
├── src/ — pipeline source (preprocessing, features, train, evaluate, predict)
├── run.py — end-to-end pipeline entry point
├── score.py — provided validator / chart renderer (unmodified)
├── outputs/ — predictions, figures, reports (generated)
├── models/ — saved model + feature metadata (generated)
└── requirements.txt


## Approach summary

See `Spotter_ML_Assessment_Report.pdf` for the full write-up. Short version:
- **Split:** chronological (train on earlier dates, hold out the most recent month) — avoids leaking future dates into training, which a random split would do.
- **Model:** LightGBM + XGBoost ensemble, Optuna-tuned, blended ~60/40.
- **Result:** RMSE $581.41, MAPE 5.28%, R² 0.85 on the holdout.
- **Known limitation:** underpredicts the rare (<0.5% of data) high-value load segment; documented in the report with a proposed fix (quantile regression / oversampling).

## Original assessment brief

See `ASSESSMENT.md` (renamed from the original `README.md`) and
`Freight_Rate_ML_Assessment.pdf` for the original task instructions.