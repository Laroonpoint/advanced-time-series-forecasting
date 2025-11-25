# Advanced Time Series Forecasting with Deep Learning and Attention

## Project Overview
This project implements a reproducible workflow for multivariate time series forecasting:
- Programmatically generates a synthetic multivariate dataset (5 features, 5000 observations) that exhibits trend, multiple seasonalities, and noise.
- Implements two models: a baseline Vanilla LSTM and an LSTM augmented with a Self-Attention mechanism.
- Trains both models, evaluates them on a held-out test set, and saves performance metrics and attention weights for interpretation.

## Files
- `generate_data.py` — generates `data.csv` (5000 x 5).
- `model.py` — model definitions for `LSTMWithAttention` and `VanillaLSTM`.
- `utils.py` — helpers for sequence creation and metrics.
- `train.py` — training script; performs a short demonstration training run and saves `results.json`, `best_model.pth`, and `attention_weights.npy`.
- `evaluate.py` — loads results and saves an attention weights image if available.
- `report.md` — this file.
- `requirements.txt` — Python package requirements.

## How to run
1. (Recommended) Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate data:
   ```bash
   python generate_data.py
   ```
   This creates `data.csv`.
3. Train and evaluate:
   ```bash
   python train.py
   ```
   This runs a short demonstration training (5 epochs per model) and saves `results.json`.
4. Produce evaluation artifacts:
   ```bash
   python evaluate.py
   ```
   This creates `attention_weights.png` (if attention model was trained).

## Notes and suggestions for extension
- The provided training run is intentionally short to keep runtime small. For production-quality results:
  - Increase epochs (50–200), larger hidden sizes, use GPU.
  - Use more systematic hyperparameter search (Optuna, KerasTuner).
  - For a rigorous baseline, compare to statistical models (ARIMA/Prophet) and multivariate variants (VAR).
- For multistep forecasting, adapt the target sequence generation and model head.
- For deployment, save the scaler and model, add a prediction API.

## Results (demo run)
See `results.json` for RMSE/MAE/MAPE for the demo training run.

