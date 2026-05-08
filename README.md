# InventIQ 🧠📦

### AI-Driven Inventory Forecasting & Optimization using LSTM + Reinforcement Learning

InventIQ is an end-to-end intelligent inventory management system built on the Walmart M5 dataset. It combines:

* **LSTM demand forecasting** for predicting future product demand
* **PPO reinforcement learning** for inventory replenishment decisions
* **SHAP explainability** for transparent AI-driven decisions
* **Gemini-powered natural language insights**
* **FastAPI + Streamlit dashboard** for real-time interaction and visualization

The system is fully CPU-compatible and designed as a practical AI-driven inventory optimization pipeline.

---

# Architecture Overview

```text
Walmart M5 Dataset
        │
        ▼
┌────────────────────┐
│ Data Pipeline      │
│ Feature Engineering│
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ LSTM Forecaster    │
│ 28-day history     │
│ → next-day demand  │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ Recursive Forecast │
│ Generate 7-day     │
│ future trajectory  │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ PPO Inventory Agent│
│ Learns reorder     │
│ quantities         │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ SHAP Explainability│
│ Why was this       │
│ decision taken?    │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ Gemini AI Insights │
│ Human-readable     │
│ inventory analysis │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ FastAPI + Streamlit│
│ Dashboard & APIs   │
└────────────────────┘
```

---

# Features

* Demand forecasting using LSTM
* Recursive 7-day forecasting
* PPO-based inventory optimization
* Inventory simulation environment using Gymnasium
* SHAP explanations for RL decisions
* Gemini-generated AI explanations
* FastAPI backend
* Streamlit interactive dashboard
* CPU-friendly training and inference

---

# Project Structure

```text
inventiq/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── app.py
│
├── outputs/
│   ├── models/
│   └── shap/
│
├── scripts/
│   ├── train.py
│   ├── knowledge.py
│   └── run_api.py
│
├── src/
│   ├── agent/
│   │   └── rl_agent.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── models.py
│   │
│   ├── data/
│   │   └── data_loader.py
│   │
│   ├── environment/
│   │   └── inventory_env.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   └── forecasting/
│       └── forecasting.py
│
├── requirements.txt
├── config.yaml
└── README.md
```

---

# Dataset

This project uses the Walmart M5 Forecasting dataset from Kaggle.

Dataset:

* `sales_train_evaluation.csv`
* `calendar.csv`
* `sell_prices.csv`

Download from:

[Walmart M5 Forecasting Dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

Place files inside:

```text
data/raw/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/VedikaSawant/inventiq.git
cd inventiq
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here

MODEL_CHECKPOINT=outputs/models/best_model.pt
PPO_CHECKPOINT=ppo_inventory.pt

SCALER_PATH=outputs/models/scaler.pkl
TARGET_SCALER_PATH=outputs/models/target_scaler.pkl

PROCESSED_DATA=data/processed/m5_processed.csv
```

---

# Training Pipeline

## Step 1 — Train Forecasting + PPO Models

```bash
python -m scripts.train
```

This performs:

* data preprocessing
* feature engineering
* LSTM training
* PPO training
* checkpoint saving

Saved models:

* `best_model.pt`
* `ppo_inventory.pt`
* `scaler.pkl`
* `target_scaler.pkl`

---

## Optional Flags

Skip preprocessing:

```bash
python -m scripts.train --skip_data
```

Skip LSTM retraining:

```bash
python -m scripts.train --skip_lstm
```

Retrain PPO only:

```bash
python -m scripts.train --skip_lstm --skip_data
```

---

# Run API Server

```bash
python -m scripts.run_api
```

API available at:

```text
http://127.0.0.1:8000
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

---

# Run Streamlit Dashboard

```bash
streamlit run frontend/app.py
```

---

# Forecasting Pipeline

The LSTM model uses:

* 28-day input sequence
* 17 engineered features
* single-step forecasting

During inference:

* recursive forecasting generates 7 future days.

---

# Forecasting Features

```text
sell_price
wday
month
is_event
is_snap

lag_1
lag_7
lag_14
lag_28

rolling_7
rolling_14
rolling_28

rolling_7_std

days_since_sale
nonzero_7
nonzero_28

item_idx
```

---

# PPO Inventory Environment

The PPO agent observes:

```text
[
    stock_norm,
    forecast_norm,
    days_since_order,
    stockout_streak
]
```

Actions:

```text
[0, 5, 10, 15, 20]
```

represent reorder quantities.

---

# Reward Function

The environment optimizes:

* low stockouts
* low overstock
* minimal holding costs
* realistic replenishment behavior

---

# Explainability

## Forecast SHAP

Explains:

* which historical features influenced demand forecasts

## PPO SHAP

Explains:

* why a specific reorder quantity was recommended

Example:

```text
High forecast demand + low stock
→ larger reorder recommendation
```

---

# API Endpoints

| Method | Endpoint            | Description                     |
| ------ | ------------------- | ------------------------------- |
| GET    | `/health`           | Health check                    |
| POST   | `/forecast`         | Generate 7-day demand forecast  |
| POST   | `/inventory`        | Inventory optimization decision |
| POST   | `/decision/explain` | SHAP explanations               |
| POST   | `/simulation`       | Inventory simulation            |

---

# Evaluation Metrics

Forecasting metrics used:

* RMSE
* MAE
* R² Score
* NRMSE
* RMSSE

RMSSE is used because it is scale-independent and standard for M5-style forecasting evaluation.

---

# Tech Stack

* PyTorch
* Gymnasium
* SHAP
* FastAPI
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Gemini API

---

# Key Design Decisions

| Decision                    | Reason                                     |
| --------------------------- | ------------------------------------------ |
| LSTM instead of Transformer | Faster CPU training                        |
| PPO for inventory control   | Handles sequential decision-making         |
| Recursive forecasting       | Enables 7-day forecasts without retraining |
| SHAP explainability         | Transparent AI decisions                   |
| CPU-first design            | Easy reproducibility                       |

---

# Future Improvements

* Multi-horizon forecasting models
* Temporal Fusion Transformer (TFT)
* Multi-store inventory optimization
* Real-time streaming inventory updates
* Supplier lead-time modeling
* Multi-agent reinforcement learning

---

# License

MIT License

---

# Acknowledgements

* Walmart M5 Forecasting Competition
* PyTorch
* SHAP
* Gymnasium
* FastAPI
* Streamlit