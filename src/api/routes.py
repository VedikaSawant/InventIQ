# src/api/routes.py

import logging
import numpy as np
import pandas as pd
import torch

from fastapi import APIRouter, HTTPException, Request

from src.api.models import (
    ForecastRequest,
    ForecastResponse,
    StockUpdateRequest,
    InventoryStatusResponse,
    DecisionRequest,
    DecisionResponse,
    SHAPFeatureImportance,
    SimulationRequest,
    SimulationResponse,
    DailySimStep,
)

from src.data.data_loader import FEATURE_COLS, SEQ_LEN
from src.environment.inventory_env import ORDER_STEP

logger = logging.getLogger(__name__)

router = APIRouter()

# =========================================================
# SHARED HELPERS
# =========================================================

def _inverse_sales(
    scaled_forecast,
    target_scaler
):

    raw = np.expm1(
        target_scaler.inverse_transform(
            scaled_forecast.reshape(-1, 1)
        ).flatten()
    )

    return np.maximum(raw, 0.0)

def _recursive_forecast(
    ctx,
    scaled_window,
    horizon=7,
):

    current_window = scaled_window.copy()

    forecasts = []

    for _ in range(horizon):

        tensor = torch.from_numpy(
            current_window
        ).float()

        scaled_fc = ctx.trainer.forecast(tensor)

        pred_scaled = float(
            scaled_fc.numpy().flatten()[0]
        )

        pred_raw = float(
            _inverse_sales(
                np.array([pred_scaled]),
                ctx.target_scaler
            )[0]
        )

        forecasts.append(pred_raw)

        # -------------------------------------------------
        # UPDATE WINDOW
        # -------------------------------------------------

        next_row = current_window[-1].copy()

        # lag_1 index
        next_row[5] = pred_scaled

        current_window = np.vstack([
            current_window[1:],
            next_row
        ])

    return np.array(forecasts)


# =========================================================
# FORECAST ROUTES
# =========================================================

@router.post("/forecast", response_model=ForecastResponse)
def run_forecast(body: ForecastRequest, request: Request):

    ctx = request.app.state.ctx

    window_np = np.array(body.window, dtype=np.float32)

    scaled = ctx.scaler.transform(window_np)

    raw_fc = _recursive_forecast(
        ctx,
        scaled,
        horizon=7
    )

    return ForecastResponse(
        item_id=body.item_id,
        forecast_units=raw_fc.tolist(),
        model_version=ctx.model_version,
    )


# =========================================================
# INVENTORY ROUTES
# =========================================================

_stock_registry = {}

@router.post("/inventory/update", response_model=InventoryStatusResponse)
def update_stock(body: StockUpdateRequest, request: Request):

    _stock_registry[body.item_id] = {
        "stock": body.current_stock,
        "date": body.date,
    }

    return get_inventory_status(body.item_id, request)


@router.get("/inventory/{item_id}", response_model=InventoryStatusResponse)
def get_inventory_status(item_id: str, request: Request):

    ctx = request.app.state.ctx

    entry = _stock_registry.get(item_id, {"stock": 0.0, "date": "unknown"})

    forecast_units = []

    try:

        df = pd.read_csv(ctx.processed_data_path)

        item_df = df[df["id"] == item_id].tail(SEQ_LEN)

        window_np = item_df[FEATURE_COLS].values.astype(np.float32)

        scaled = ctx.scaler.transform(window_np)

        raw_fc = _recursive_forecast(
            ctx,
            scaled,
            horizon=7
        )

        forecast_units = raw_fc.tolist()

    except Exception:
        pass

    avg = np.mean(forecast_units) if forecast_units else 1

    days = entry["stock"] / avg

    reorder_flag = days < 3

    return InventoryStatusResponse(
        item_id=item_id,
        current_stock=entry["stock"],
        date=entry["date"],
        reorder_flag=reorder_flag,
        days_of_coverage=round(days, 2),
        forecast_units=forecast_units,
    )


# =========================================================
# DECISION ROUTES
# =========================================================

_last_decision = {}

@router.post("/decisions/recommend", response_model=DecisionResponse)
def recommend_order(body: DecisionRequest, request: Request):

    ctx = request.app.state.ctx

    window_np = np.array(body.window, dtype=np.float32)

    scaled = ctx.scaler.transform(window_np)

    raw_fc = _recursive_forecast(
        ctx,
        scaled,
        horizon=7
    )

    next_day_forecast = raw_fc[0]

    obs = np.array([
        body.current_stock / 500.0,
        next_day_forecast / 500.0,
        0.0,
        0.0,
    ], dtype=np.float32)

    action = ctx.ppo_agent.predict(obs)

    order_qty = action * ORDER_STEP

    explanation = ""

    shap_importances = []

    if ctx.agent_explainer:

        shap_result = ctx.agent_explainer.explain(
            obs=obs,
            action_taken=action,
            order_qty=order_qty,
            item_id=body.item_id,
        )

        explanation = shap_result["natural_language_summary"]

        shap_importances = [
            SHAPFeatureImportance(
                feature=k,
                value=round(abs(v), 4),
                direction="increases_order" if v > 0 else "decreases_order",
            )
            for k, v in shap_result["feature_importances"].items()
        ]

    return DecisionResponse(
        item_id=body.item_id,
        recommended_order=order_qty,
        action_index=action,
        forecast_units=raw_fc.tolist(),
        shap_importances=shap_importances,
        explanation_summary=explanation,
    )


# =========================================================
# SIMULATION ROUTES
# =========================================================

@router.post("/simulation/run", response_model=SimulationResponse)
def run_simulation(body: SimulationRequest, request: Request):

    ctx = request.app.state.ctx

    from src.environment.inventory_env import InventoryEnv

    demand_series = np.random.randint(10, 30, body.n_days)

    feature_matrix = np.random.rand(body.n_days, len(FEATURE_COLS))

    env = InventoryEnv(
        demand_series=demand_series,
        feature_matrix=feature_matrix,
        forecaster=ctx.trainer.model,
        scaler=ctx.scaler,
        target_scaler=ctx.target_scaler,
        initial_stock=body.initial_stock,
    )

    obs, _ = env.reset()

    steps = []

    total_reward = 0

    for day in range(body.n_days):

        action = ctx.ppo_agent.predict(obs)

        obs, reward, done, _, info = env.step(action)

        total_reward += reward

        steps.append(
            DailySimStep(
                day=day + 1,
                stock=info["stock"],
                demand=info["demand_today"],
                order_qty=info["order_qty"],
                units_sold=info["demand_today"] - info["unmet_demand"],
                unmet_demand=info["unmet_demand"],
                reward=reward,
            )
        )

        if done:
            break

    return SimulationResponse(
        item_id=body.item_id,
        policy=body.policy,
        total_reward=total_reward,
        service_level=1.0,
        mean_stock=0,
        stockout_days=0,
        steps=steps,
    )