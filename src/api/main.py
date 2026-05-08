# src/api/main.py

from dotenv import load_dotenv
load_dotenv()

import logging
import os
import pickle
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.forecasting.forecasting import (
    DemandForecastingSystem
)

from src.agent.rl_agent import PPOAgent

from src.explainability.shap_explainer import (
    build_agent_explainer,
)

from src.api.routes import router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# ENVIRONMENT CONFIG
# =========================================================

MODEL_CHECKPOINT = os.getenv(
    "MODEL_CHECKPOINT",
    "outputs/models/best_model.pt"
)

PPO_CHECKPOINT = os.getenv(
    "PPO_CHECKPOINT",
    "ppo_inventory.pt"
)

SCALER_PATH = os.getenv(
    "SCALER_PATH",
    "outputs/models/scaler.pkl"
)

TARGET_SCALER_PATH = os.getenv(
    "TARGET_SCALER_PATH",
    "outputs/models/target_scaler.pkl"
)

PROCESSED_DATA = os.getenv(
    "PROCESSED_DATA",
    "data/processed/m5_processed.csv"
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "v1.0"
)


# =========================================================
# APP STATE OBJECT
# =========================================================

class AppState:

    trainer = None

    ppo_agent = None

    scaler = None

    target_scaler = None

    agent_explainer = None

    model_version = None

    processed_data_path = None


# =========================================================
# STARTUP / SHUTDOWN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    ctx = AppState()

    # -----------------------------------------------------
    # 1️⃣ Load Forecasting System
    # -----------------------------------------------------

    logger.info("Loading forecasting system...")

    ctx.trainer = DemandForecastingSystem()

    ctx.trainer.load()

    ctx.trainer.model.eval()

    logger.info("Forecasting model loaded ✓")


    # -----------------------------------------------------
    # 2️⃣ Load Feature Scaler
    # -----------------------------------------------------

    logger.info(f"Loading scaler from {SCALER_PATH}...")

    with open(SCALER_PATH, "rb") as f:
        ctx.scaler = pickle.load(f)

    logger.info("Feature scaler loaded ✓")


    # -----------------------------------------------------
    # 3️⃣ Load Target Scaler
    # -----------------------------------------------------

    logger.info(
        f"Loading target scaler from {TARGET_SCALER_PATH}..."
    )

    with open(TARGET_SCALER_PATH, "rb") as f:
        ctx.target_scaler = pickle.load(f)

    logger.info("Target scaler loaded ✓")


    # -----------------------------------------------------
    # 4️⃣ Load PPO Agent
    # -----------------------------------------------------

    logger.info("Loading PPO agent...")

    ctx.ppo_agent = PPOAgent(
        state_dim=4,
        n_actions=5
    )

    ctx.ppo_agent.load(PPO_CHECKPOINT)

    ctx.ppo_agent.policy.eval()

    logger.info("PPO agent loaded ✓")


    # -----------------------------------------------------
    # 5️⃣ Load SHAP Explainer (Optional)
    # -----------------------------------------------------

    ctx.agent_explainer = None

    try:

        rollout_path = "outputs/models/rollout_states.npy"

        if os.path.exists(rollout_path):

            rollout_states = np.load(rollout_path)

            ctx.agent_explainer = build_agent_explainer(
                ctx.ppo_agent,
                rollout_states
            )

            logger.info("SHAP explainer loaded ✓")

        else:

            logger.warning(
                f"Rollout states not found at "
                f"{rollout_path} — SHAP disabled"
            )

    except Exception as e:

        logger.warning(
            f"SHAP explainer init failed "
            f"(non-fatal): {e}"
        )


    # -----------------------------------------------------
    # Store metadata
    # -----------------------------------------------------

    ctx.model_version = MODEL_VERSION

    ctx.processed_data_path = PROCESSED_DATA


    # -----------------------------------------------------
    # Attach app state
    # -----------------------------------------------------

    app.state.ctx = ctx

    logger.info("InventIQ API startup complete ✓")

    yield

    logger.info("InventIQ API shutting down.")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="InventIQ API",
    description=(
        "AI-Driven Inventory Optimization "
        "— Forecast · Decide · Explain"
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROUTERS
# =========================================================

app.include_router(router)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health", tags=["Health"])
def health():

    return {
        "status": "ok",
        "version": MODEL_VERSION,
    }