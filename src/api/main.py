# src/api/main.py

from dotenv import load_dotenv
load_dotenv()

import logging
import os
import pickle
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.forecasting.model import DemandLSTM
from src.forecasting.trainer import LSTMTrainer
from src.agent.ppo_agent import PPOAgent

from src.explainability.shap_explainer import (
    build_agent_explainer,
)

from src.knowledge.vector_store import VectorStore

# ✅ Updated router imports
from src.api.routes import router
from src.api.chat import router as chat_router


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
    "outputs/models/ppo_inventory.pt"
)

SCALER_PATH = os.getenv(
    "SCALER_PATH",
    "outputs/models/scaler.pkl"
)

PROCESSED_DATA = os.getenv(
    "PROCESSED_DATA",
    "data/processed/m5_processed.csv"
)

VECTOR_STORE_DIR = os.getenv(
    "VECTOR_STORE_DIR",
    "outputs/vector_store"
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

    forecast_explainer = None
    agent_explainer = None

    vector_store = None

    model_version = None
    processed_data_path = None


# =========================================================
# STARTUP / SHUTDOWN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    ctx = AppState()

    # -----------------------------------------------------
    # 1️⃣ Load LSTM Forecaster
    # -----------------------------------------------------

    logger.info("Loading DemandLSTM...")

    lstm_model = DemandLSTM()

    ctx.trainer = LSTMTrainer(
        lstm_model,
        checkpoint_dir="outputs/models"
    )

    ctx.trainer.load("best_model.pt")

    ctx.trainer.model.eval()


    # -----------------------------------------------------
    # 2️⃣ Load StandardScaler
    # -----------------------------------------------------

    logger.info(f"Loading scaler from {SCALER_PATH}...")

    with open(SCALER_PATH, "rb") as f:
        ctx.scaler = pickle.load(f)


    # -----------------------------------------------------
    # 3️⃣ Load PPO Agent
    # -----------------------------------------------------

    logger.info("Loading PPO agent...")

    ctx.ppo_agent = PPOAgent(
        state_dim=10,
        n_actions=11
    )

    ctx.ppo_agent.load("ppo_inventory.pt")

    ctx.ppo_agent.policy.eval()


    # -----------------------------------------------------
    # 4️⃣ Load SHAP Explainer (Optional)
    # -----------------------------------------------------

    ctx.agent_explainer = None

    try:

        rollout_path = "outputs/models/rollout_states.npy"

        if os.path.exists(rollout_path):

            import numpy as np

            rollout_states = np.load(rollout_path)

            ctx.agent_explainer = build_agent_explainer(
                ctx.ppo_agent,
                rollout_states
            )

            logger.info("AgentExplainer loaded ✓")

        else:

            logger.warning(
                f"Rollout states not found at {rollout_path} — SHAP disabled"
            )

    except Exception as e:

        logger.warning(
            f"SHAP explainer init failed (non-fatal): {e}"
        )


    # -----------------------------------------------------
    # 5️⃣ Load Vector Store
    # -----------------------------------------------------

    logger.info("Connecting to VectorStore...")

    ctx.vector_store = VectorStore(
        persist_dir=VECTOR_STORE_DIR
    )


    # -----------------------------------------------------
    # Store metadata
    # -----------------------------------------------------

    ctx.model_version = MODEL_VERSION

    ctx.processed_data_path = PROCESSED_DATA


    # Attach context

    app.state.ctx = ctx

    logger.info("InventIQ API startup complete ✓")

    yield

    logger.info("InventIQ API shutting down.")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="InventIQ API",
    description="AI-Driven Inventory Optimization — Forecast · Decide · Explain",
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

# ✅ merged standard routes
app.include_router(router)

# ✅ streaming assistant
app.include_router(
    chat_router,
    prefix="/chat",
    tags=["Chat"]
)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health", tags=["Health"])
def health():

    return {
        "status": "ok",
        "version": MODEL_VERSION
    }