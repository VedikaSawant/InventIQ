"""
src/api/schemas.py
-------------------
All Pydantic request/response models for the InventIQ API.

Kept in one file so every router imports from a single source of truth.
Naming convention:
    <Resource>Request   — incoming POST body
    <Resource>Response  — outgoing JSON
    <Resource>Item      — nested model used inside a response
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ── Shared primitives ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str


# ── Forecast ──────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    """
    Raw feature window for a single SKU.
    The API scales internally — caller sends raw values.
    """
    item_id: str = Field(..., description="SKU identifier, e.g. HOBBIES_1_001_CA_1")
    window:  list[list[float]] = Field(
        ...,
        description=(
            "28 × 9 feature matrix (rows=days, cols=features in FEATURE_COLS order): "
            "sales, sell_price, wday, month, is_event, is_snap, lag_7, lag_28, rolling_7"
        ),
        min_length=28,
        max_length=28,
    )


class ForecastResponse(BaseModel):
    item_id:          str
    forecast_units:   list[float] = Field(..., description="Predicted demand for days +1…+7")
    horizon_days:     int         = 7
    model_version:    str


# ── Inventory ─────────────────────────────────────────────────────────────────

class StockUpdateRequest(BaseModel):
    item_id:       str
    current_stock: float = Field(..., ge=0)
    date:          str   = Field(..., description="ISO date string, e.g. 2016-01-15")


class InventoryStatusResponse(BaseModel):
    item_id:          str
    current_stock:    float
    date:             str
    reorder_flag:     bool  = Field(..., description="True if stock ≤ reorder threshold")
    days_of_coverage: float = Field(..., description="Estimated days before stockout")
    forecast_units:   list[float]


# ── Decisions ─────────────────────────────────────────────────────────────────

class DecisionRequest(BaseModel):
    item_id:       str
    current_stock: float = Field(..., ge=0)
    window:        list[list[float]] = Field(..., min_length=28, max_length=28)


class SHAPFeatureImportance(BaseModel):
    feature: str
    value:   float
    direction: str = Field(..., description="'increases_order' or 'decreases_order'")


class DecisionResponse(BaseModel):
    item_id:             str
    recommended_order:   int   = Field(..., description="Units to order (0–100)")
    action_index:        int   = Field(..., description="Discrete action 0–10")
    forecast_units:      list[float]
    shap_importances:    list[SHAPFeatureImportance]
    explanation_summary: str
    plot_path:           str | None = None


# ── Simulation ────────────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    item_id:          str
    initial_stock:    float = Field(50.0,  ge=0)
    holding_cost:     float = Field(0.5,   gt=0)
    stockout_penalty: float = Field(2.0,   gt=0)
    n_days:           int   = Field(30,    ge=7, le=365)
    policy:           str   = Field(
        "ppo",
        description="One of: ppo | eoq | reorder | forecast"
    )
    # EOQ / reorder params (optional overrides)
    reorder_point:    float | None = None
    max_level:        float | None = None


class DailySimStep(BaseModel):
    day:          int
    stock:        float
    demand:       float
    order_qty:    int
    units_sold:   float
    unmet_demand: float
    reward:       float


class SimulationResponse(BaseModel):
    item_id:           str
    policy:            str
    total_reward:      float
    service_level:     float = Field(..., description="Fraction of demand met")
    mean_stock:        float
    stockout_days:     int
    steps:             list[DailySimStep]


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str  = Field(..., min_length=1)
    item_id:    str | None = Field(None, description="Scope retrieval to a specific SKU")
    session_id: str | None = None           # for multi-turn history (future use)


# Chat response is streamed as SSE — no response model needed for the endpoint,
# but we define the event payload shape for documentation:
class ChatEventPayload(BaseModel):
    token:      str | None = None           # streaming token
    done:       bool       = False
    sources:    list[str]  = []             # retrieved chunk excerpts, sent on done=True
    session_id: str | None = None