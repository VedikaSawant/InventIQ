# frontend/app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

from pathlib import Path
import matplotlib.pyplot as plt

import google.generativeai as genai
import os

from src.knowledge.vector_store import VectorStore
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(env_path)

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

vector_store = VectorStore()

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="InventIQ Dashboard",
    layout="wide"
)

API_URL = "http://localhost:8000/chat/"

DATA_PATH = "data/processed/m5_processed.csv"

SHAP_DIR = Path("outputs/shap")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    if not Path(DATA_PATH).exists():

        st.error(
            f"Processed data not found at {DATA_PATH}"
        )

        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    return df


df = load_data()

if df.empty:

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("InventIQ")

page = st.sidebar.radio(

    "Navigation",

    [
        "🏪 Dashboard",

        "📦 Forecast",

        "🤖 Decisions",

        "✨ AI Insights"
    ]
)


# Select Item

ITEM_COL = "item_id"

if ITEM_COL in df.columns:

    items = sorted(df[ITEM_COL].unique())

    selected_item = st.sidebar.selectbox(

        "Select Item",

        items

    )

else:

    selected_item = None

# =========================================================
# DASHBOARD
# =========================================================

if page == "🏪 Dashboard":

    st.title("🏪 Inventory Dashboard")

    if not selected_item:

        st.warning("Please select an item.")
        st.stop()

    try:

        # -------------------------------------------------
        # Step 1 — Get Inventory Status
        # -------------------------------------------------

        inv_response = requests.get(
            f"http://localhost:8000/inventory/{selected_item}"
        )

        if inv_response.status_code != 200:

            st.error(
                f"Inventory API failed: {inv_response.text}"
            )

            st.stop()

        inv_data = inv_response.json()

        forecast_values = inv_data.get(
            "forecast_units", []
        )

        current_stock = inv_data.get(
            "current_stock", 0
        )

        days_of_coverage = inv_data.get(
            "days_of_coverage", 0
        )

        reorder_flag = inv_data.get(
            "reorder_flag", False
        )

        # -------------------------------------------------
        # Step 2 — Build window for decision API
        # -------------------------------------------------

        item_df = df[
            df[ITEM_COL] == selected_item
        ]

        window = item_df.tail(28)[[
            "sales",
            "sell_price",
            "wday",
            "month",
            "is_event",
            "is_snap",
            "lag_7",
            "lag_28",
            "rolling_7"
        ]].values.tolist()

        if len(window) < 28:

            st.error("Window must contain 28 rows")

            st.stop()

        decision_payload = {
            "item_id": selected_item,
            "current_stock": current_stock,
            "window": window
        }

        decision_response = requests.post(
            "http://localhost:8000/decisions/recommend",
            json=decision_payload
        )

        if decision_response.status_code != 200:

            st.error(
                f"Decision API failed: {decision_response.text}"
            )

            st.stop()

        decision_data = decision_response.json()

        recommended_order = decision_data.get(
            "recommended_order", 0
        )

        shap_data = decision_data.get(
            "shap_importances", []
        )

    except Exception as e:

        st.error(f"API error: {e}")

        st.stop()

    # -------------------------------------------------
    # Forecast Plot
    # -------------------------------------------------

    if forecast_values:

        st.subheader(
            "📈 7-Day Forecast"
        )

        days = np.arange(
            1,
            len(forecast_values) + 1
        )

        fig, ax = plt.subplots()

        ax.plot(
            days,
            forecast_values
        )

        ax.set_xlabel("Day")

        ax.set_ylabel("Demand")

        ax.set_title(
            "Upcoming Demand Forecast"
        )

        st.pyplot(fig)

        # Optional forecast table

        forecast_df = pd.DataFrame({
            "Day": days,
            "Forecast Units": forecast_values
        })

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

    # -------------------------------------------------
    # Recent Demand Trend
    # -------------------------------------------------

    st.subheader(
        "📊 Last 30 Days Demand"
    )

    recent_sales = item_df["sales"].tail(30)

    fig2, ax2 = plt.subplots()

    ax2.plot(
        recent_sales.values
    )

    ax2.set_title(
        "Recent Demand Trend"
    )

    ax2.set_xlabel("Day")

    ax2.set_ylabel("Units Sold")

    st.pyplot(fig2)

    # -------------------------------------------------
    # Top Selling Items
    # -------------------------------------------------

    st.subheader(
        "🏆 Top Selling Items"
    )

    top_items = (
        df.groupby(ITEM_COL)["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.dataframe(
        top_items.reset_index(),
        use_container_width=True
    )


# =========================================================
# FORECAST
# =========================================================

elif page == "📦 Forecast":

    st.title("📦 Forecast View")

    st.info("Forecast data comes from trained LSTM model.")

    # -------------------------------------------------
    # Top Selling Items Table
    # -------------------------------------------------

    st.subheader("🔥 Top Selling Items")

    top_items = (
        df.groupby("item_id")["sales"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(
        top_items,
        use_container_width=True
    )

    forecast_days = np.arange(1, 8)

    # -------------------------------------------------
    # Filter selected item
    # -------------------------------------------------

    item_df = df[df["item_id"] == selected_item]

    if len(item_df) < 28:

        st.error("Not enough history to generate forecast.")
        st.stop()

    # -------------------------------------------------
    # Build correct window (VERY IMPORTANT)
    # Must match training features exactly
    # -------------------------------------------------

    window = item_df.tail(28)[[
        "sales",
        "sell_price",
        "wday",
        "month",
        "is_event",
        "is_snap",
        "lag_7",
        "lag_28",
        "rolling_7"
    ]].values.tolist()

    st.write("Window length:", len(window))  # Debug check

    payload = {
        "item_id": selected_item,
        "window": window
    }

    # -------------------------------------------------
    # Call Forecast API
    # -------------------------------------------------

    try:

        response = requests.post(
            "http://localhost:8000/forecast",
            json=payload
        )

        if response.status_code != 200:

            st.error(
                f"API request failed: {response.text}"
            )

            st.stop()

        data = response.json()

        if "forecast_units" not in data:

            st.error(
                f"Unexpected API response: {data}"
            )

            st.stop()

        forecast_values = data["forecast_units"]

        # -------------------------------------------------
        # Plot forecast
        # -------------------------------------------------

        fig, ax = plt.subplots()

        ax.plot(
            forecast_days,
            forecast_values
        )

        ax.set_title("7-Day Forecast")

        ax.set_xlabel("Day")

        ax.set_ylabel("Demand")

        st.pyplot(fig)

    except Exception as e:

        st.error(f"Forecast API error: {e}")

# =========================================================
# DECISIONS PAGE
# =========================================================

elif page == "🤖 Decisions":

    st.title("🤖 AI Decision Output")

    if not selected_item:

        st.warning("Please select an item.")
        st.stop()

    try:

        # ---------------------------------------------
        # Build window from data
        # ---------------------------------------------

        item_df = df[
            df[ITEM_COL] == selected_item
        ]

        window = item_df.tail(28)[[
            "sales",
            "sell_price",
            "wday",
            "month",
            "is_event",
            "is_snap",
            "lag_7",
            "lag_28",
            "rolling_7"
        ]].values.tolist()

        if len(window) < 28:

            st.error(
                "Not enough history for decision."
            )

            st.stop()

        # ---------------------------------------------
        # Get inventory info
        # ---------------------------------------------

        inv_response = requests.get(
            f"http://localhost:8000/inventory/{selected_item}"
        )

        inv_data = inv_response.json()

        current_stock = inv_data.get(
            "current_stock",
            0
        )

        # ---------------------------------------------
        # Call Decision API
        # ---------------------------------------------

        payload = {
            "item_id": selected_item,
            "current_stock": current_stock,
            "window": window
        }

        response = requests.post(
            "http://localhost:8000/decisions/recommend",
            json=payload
        )

        if response.status_code != 200:

            st.error(
                f"Decision API failed: {response.text}"
            )

            st.stop()

        decision_data = response.json()

        recommended_order = decision_data.get(
            "recommended_order",
            0
        )

        forecast_values = decision_data.get(
            "forecast_units",
            []
        )

        shap_data = decision_data.get(
            "shap_importances",
            []
        )

        explanation_text = decision_data.get(
            "explanation_summary",
            ""
        )

    except Exception as e:

        st.error(f"API error: {e}")
        st.stop()

    # ---------------------------------------------
    # KPI Display
    # ---------------------------------------------

    col1, col2 = st.columns(2)

    col1.metric(
        "📦 Current Stock",
        current_stock
    )

    col2.metric(
        "🛒 Recommended Order",
        recommended_order
    )

    # ---------------------------------------------
    # Forecast Plot
    # ---------------------------------------------

    if forecast_values:

        st.subheader(
            "📈 Forecast Used for Decision"
        )

        days = np.arange(
            1,
            len(forecast_values) + 1
        )

        fig, ax = plt.subplots()

        ax.plot(
            days,
            forecast_values
        )

        ax.set_title(
            "Forecast Demand"
        )

        ax.set_xlabel("Day")

        ax.set_ylabel("Units")

        st.pyplot(fig)

    # ---------------------------------------------
    # Explanation Text
    # ---------------------------------------------

    if explanation_text:

        st.subheader(
            "📝 Decision Explanation"
        )

        st.info(
            explanation_text
        )

    # ---------------------------------------------
    # SHAP Plot (Moved from Explainability page)
    # ---------------------------------------------

    if shap_data:

        st.subheader("🔍 Feature Contributions")

        shap_df = pd.DataFrame(
            shap_data
        )

        shap_df = shap_df.sort_values(
            "value",
            ascending=False
        )

        fig, ax = plt.subplots()

        ax.barh(
            shap_df["feature"],
            shap_df["value"]
        )

        ax.set_title(
            "SHAP Feature Contributions"
        )

        ax.set_xlabel("Impact on Order")

        st.pyplot(fig)

    else:

        st.warning(
            "No SHAP data available."
        )


# =========================================================
# AI INSIGHTS PAGE (SHAP + EMBEDDINGS + GEMINI)
# =========================================================

elif page == "✨ AI Insights":

    st.title("✨ AI Insights")

    if not selected_item:

        st.warning("Please select an item.")
        st.stop()

    try:

        # --------------------------------------------
        # Build window for decision API
        # --------------------------------------------

        item_df = df[
            df[ITEM_COL] == selected_item
        ]

        window = item_df.tail(28)[[
            "sales",
            "sell_price",
            "wday",
            "month",
            "is_event",
            "is_snap",
            "lag_7",
            "lag_28",
            "rolling_7"
        ]].values.tolist()

        payload = {
            "item_id": selected_item,
            "current_stock": 0,
            "window": window
        }

        decision_response = requests.post(
            "http://localhost:8000/decisions/recommend",
            json=payload
        )

        decision_data = decision_response.json()

        forecast_values = decision_data.get(
            "forecast_units", []
        )

        shap_data = decision_data.get(
            "shap_importances", []
        )

        explanation = decision_data.get(
            "explanation_summary",
            ""
        )

        recommended_order = decision_data.get(
            "recommended_order", 0
        )

    except Exception as e:

        st.error(f"Decision API error: {e}")
        st.stop()

    # --------------------------------------------
    # Retrieve Embedding Context
    # --------------------------------------------

    query_text = f"""
Explain the most recent inventory decision
for item {selected_item}.
"""

    retrieved_chunks = vector_store.query_for_item(

        query_text=query_text,

        item_id=selected_item,

        top_k=5
    )

    context_text = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    # --------------------------------------------
    # Show Raw Decision Summary
    # --------------------------------------------

    st.subheader("📌 Decision Summary")

    st.success(explanation)

    # --------------------------------------------
    # Show Top SHAP Features
    # --------------------------------------------

    if shap_data:

        shap_df = pd.DataFrame(
            shap_data
        )

        shap_df = shap_df.sort_values(
            "value",
            ascending=False
        )

        top_features = shap_df.head(3)

        st.subheader(
            "🔍 Key Influencing Features"
        )

        for _, row in top_features.iterrows():

            st.write(
                f"• **{row['feature']}** "
                f"({row['direction']})"
            )

    # --------------------------------------------
    # Gemini AI Insight (Uses Embeddings + SHAP)
    # --------------------------------------------

    st.subheader("🤖 AI Generated Insights")

    try:

        prompt = f"""
You are an AI inventory assistant.

Use the retrieved inventory knowledge
and current decision details to generate insights.

Retrieved Knowledge:
{context_text}

Current Decision:

Item: {selected_item}

Recommended Order:
{recommended_order}

Forecast:
{forecast_values}

Top SHAP Features:
{top_features.to_dict()}

Provide:

• Demand outlook  
• Why this order was recommended  
• Any stock risks  
• Practical stock planning advice  

Use short bullet points.
"""

        gemini_response = gemini_model.generate_content(
            prompt
        )

        st.info(
            gemini_response.text
        )

    except Exception as e:

        st.warning(
            f"Gemini insight failed: {e}"
        )