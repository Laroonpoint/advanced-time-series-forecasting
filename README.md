# Advanced Time Series Forecasting with Deep Learning and Attention

This project implements a complete real-world workflow for **multivariate time series forecasting** using a **PyTorch LSTM with a Self-Attention mechanism**, compared against a **baseline LSTM**.

## 🚀 Features
- Programmatically generated multivariate dataset (5 features × 5000 steps)
- Data preprocessing and scaling
- Sequence generation for supervised learning
- Two forecasting models:
  - **LSTM + Self-Attention**
  - **Vanilla LSTM**
- Hyperparameter configuration
- Evaluation using RMSE, MAE, MAPE
- Attention weight extraction & visualization
- Clear, modular Python codebase

---

## 📂 Project Structure
```
advanced_time_series_project/
│── generate_data.py
│── model.py
│── utils.py
│── train.py
│── evaluate.py
│── report.md
│── requirements.txt
│── data.csv (generated)
│── attention_weights.npy (after training)
```

---

## 🛠 Installation
```bash
pip install -r requirements.txt
```

---

## 📊 Generate Dataset
```bash
python generate_data.py
```

---

## 🧠 Train Models
```bash
python train.py
```

This trains:
- ✔ LSTM with Attention  
- ✔ Vanilla LSTM  
and saves results as:

- `results.json`
- `best_model.pth`
- `attention_weights.npy`

---

## 📈 Evaluate & Visualize
```bash
python evaluate.py
```

If attention weights are present, this generates:
- `attention_weights.png`

---

## 📌 Notes
- Increase epochs (50–200) for real accuracy.
- GPU recommended for faster training.
- You can extend this to multi-step forecasting or add other baselines (ARIMA, Prophet, VAR).

---

## 📝 License
MIT License.
