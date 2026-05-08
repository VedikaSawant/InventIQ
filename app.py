# frontend/app.py

import os
from pathlib import Path

import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


# =========================================================
# ENV
# =========================================================

env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(env_path)

KEY_1 = os.getenv("GOOGLE_API_KEY_1")
KEY_2 = os.getenv("GOOGLE_API_KEY_2")


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="InventIQ Dashboard",
    layout="wide"
)

BASE_API = "http://localhost:8000"

DATA_PATH = "data/processed/m5_processed.csv"


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

    return pd.read_csv(DATA_PATH)


df = load_data()

if df.empty:
    st.stop()


# =========================================================
# FEATURE CONFIG
# MUST MATCH TRAINING PIPELINE EXACTLY
# =========================================================

FEATURE_COLS = [
    "sell_price",

    # calendar
    "wday",
    "month",
    "is_event",
    "is_snap",

    # lag features
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    # rolling means
    "rolling_7",
    "rolling_14",
    "rolling_28",

    # volatility
    "rolling_7_std",

    # sparse demand
    "days_since_sale",
    "nonzero_7",
    "nonzero_28",

    # item identity
    "item_idx"
]

WINDOW_SIZE = 28


# =========================================================
# HELPERS
# =========================================================

def build_window(item_df):

    required = item_df.tail(WINDOW_SIZE)

    if len(required) < WINDOW_SIZE:
        return None

    return required[FEATURE_COLS].values.tolist()


def call_forecast_api(item_id, window):

    payload = {
        "item_id": item_id,
        "window": window
    }

    response = requests.post(
        f"{BASE_API}/forecast",
        json=payload
    )

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()


def call_inventory_api(item_id):

    response = requests.get(
        f"{BASE_API}/inventory/{item_id}"
    )

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()


def call_decision_api(item_id, current_stock, window):

    payload = {
        "item_id": item_id,
        "current_stock": current_stock,
        "window": window
    }

    response = requests.post(
        f"{BASE_API}/decisions/recommend",
        json=payload
    )

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()


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

ITEM_COL = "item_id"

selected_item = None

if ITEM_COL in df.columns:

    items = sorted(df[ITEM_COL].unique())

    selected_item = st.sidebar.selectbox(
        "Select Item",
        items
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏪 Dashboard":

    st.title("🏪 Inventory Dashboard")

    if not selected_item:
        st.stop()

    try:

        item_df = df[
            df[ITEM_COL] == selected_item
        ]

        window = build_window(item_df)

        if window is None:

            st.error(
                "Not enough history."
            )

            st.stop()

        inv_data = call_inventory_api(
            selected_item
        )

        current_stock = inv_data.get(
            "current_stock",
            0
        )

        days_of_coverage = inv_data.get(
            "days_of_coverage",
            0
        )

        reorder_flag = inv_data.get(
            "reorder_flag",
            False
        )

        decision_data = call_decision_api(
            selected_item,
            current_stock,
            window
        )

        forecast_values = decision_data.get(
            "forecast_units",
            []
        )

        recommended_order = decision_data.get(
            "recommended_order",
            0
        )

    except Exception as e:

        st.error(f"API error: {e}")
        st.stop()

    # -----------------------------------------------------
    # KPIs
    # -----------------------------------------------------

    c1, = st.columns(1)

    c1.metric(
        "🛒 Recommended Order",
        int(recommended_order)
    )

    if reorder_flag:
        st.warning("⚠️ Reorder Recommended")
    else:
        st.success("✅ Inventory Stable")

    # -----------------------------------------------------
    # Forecast
    # -----------------------------------------------------

    if forecast_values:

        st.subheader("📈 7-Day Forecast")

        days = np.arange(
            1,
            len(forecast_values) + 1
        )

        fig, ax = plt.subplots()

        ax.plot(days, forecast_values)

        ax.set_xlabel("Day")
        ax.set_ylabel("Demand")

        st.pyplot(fig)

    # -----------------------------------------------------
    # Demand trend
    # -----------------------------------------------------

    st.subheader("📊 Last 30 Days Demand")

    recent_sales = item_df["sales"].tail(30)

    fig2, ax2 = plt.subplots()

    ax2.plot(recent_sales.values)

    ax2.set_xlabel("Day")
    ax2.set_ylabel("Units")

    st.pyplot(fig2)

    # -----------------------------------------------------
    # Top items
    # -----------------------------------------------------

    st.subheader("🏆 Top Selling Items")

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
# FORECAST PAGE
# =========================================================

elif page == "📦 Forecast":

    st.title("📦 Forecast")

    if not selected_item:
        st.stop()

    try:

        item_df = df[
            df[ITEM_COL] == selected_item
        ]

        window = build_window(item_df)

        if window is None:

            st.error(
                "Not enough history."
            )

            st.stop()

        data = call_forecast_api(
            selected_item,
            window
        )

        forecast_values = data.get(
            "forecast_units",
            []
        )

        st.subheader("📈 Forecast")

        days = np.arange(
            1,
            len(forecast_values) + 1
        )

        fig, ax = plt.subplots()

        ax.plot(days, forecast_values)

        ax.set_xlabel("Day")
        ax.set_ylabel("Forecast Units")

        st.pyplot(fig)

        forecast_df = pd.DataFrame({
            "Day": days,
            "Forecast Units": forecast_values
        })

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

    except Exception as e:

        st.error(f"Forecast error: {e}")


# =========================================================
# DECISIONS PAGE
# =========================================================

elif page == "🤖 Decisions":

    st.title("🤖 Inventory Decisions")

    if not selected_item:
        st.stop()

    try:

        item_df = df[
            df[ITEM_COL] == selected_item
        ]

        window = build_window(item_df)

        if window is None:

            st.error(
                "Not enough history."
            )

            st.stop()

        inv_data = call_inventory_api(
            selected_item
        )

        current_stock = inv_data.get(
            "current_stock",
            0
        )

        decision_data = call_decision_api(
            selected_item,
            current_stock,
            window
        )

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

    except Exception as e:

        st.error(f"Decision API error: {e}")
        st.stop()

    # -----------------------------------------------------
    # KPI
    # -----------------------------------------------------

    st.metric(
        "🛒 Recommended Order",
        int(recommended_order)
    )

    # -----------------------------------------------------
    # Forecast
    # -----------------------------------------------------

    if forecast_values:

        st.subheader(
            "📈 Forecast Used"
        )

        days = np.arange(
            1,
            len(forecast_values) + 1
        )

        fig, ax = plt.subplots()

        ax.plot(days, forecast_values)

        ax.set_xlabel("Day")
        ax.set_ylabel("Demand")

        st.pyplot(fig)

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    if shap_data:

        st.subheader(
            "🔍 Feature Contributions"
        )

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

        ax.set_xlabel("Impact")

        st.pyplot(fig)

        # -------------------------------------------------
        # Gemini explanation
        # -------------------------------------------------

        try:

            top_features = shap_df.head(3)

            prompt = f"""
You are an inventory optimization assistant.

Explain this inventory decision.

Item:
{selected_item}

Recommended Order:
{recommended_order}

Forecast:
{forecast_values}

Top Influencing Features:
{top_features.to_dict()}

Requirements:
- Explain WHY the order was recommended
- Mention the important features
- Use business-friendly language
- Use concise bullet points
"""

            genai.configure(api_key=KEY_1)

            model = genai.GenerativeModel(
                "gemini-2.5-flash"
            )

            response = model.generate_content(
                prompt
            )

            st.subheader(
                "📝 AI Explanation"
            )

            st.success(response.text)

        except Exception as e:

            st.warning(
                f"Gemini explanation failed: {e}"
            )


# =========================================================
# AI INSIGHTS PAGE
# =========================================================

elif page == "✨ AI Insights":

    st.title("✨ AI Insights")

    if not selected_item:
        st.stop()

    try:

        item_df = df[
            df[ITEM_COL] == selected_item
        ]

        window = build_window(item_df)

        if window is None:

            st.error(
                "Not enough history."
            )

            st.stop()

        decision_data = call_decision_api(
            selected_item,
            0,
            window
        )

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

        shap_df = pd.DataFrame(
            shap_data
        )

        shap_df = shap_df.sort_values(
            "value",
            ascending=False
        )

        top_features = shap_df.head(5)

    except Exception as e:

        st.error(f"AI insight error: {e}")
        st.stop()

    st.subheader("📌 Decision Summary")

    st.write(
        f"Recommended Order: {recommended_order}"
    )

    st.subheader(
        "🔍 Key Drivers"
    )

    st.dataframe(
        top_features,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Gemini insights
    # -----------------------------------------------------

    try:

        prompt = f"""
You are an AI inventory optimization assistant.

Analyze this inventory situation.

Item:
{selected_item}

Recommended Order:
{recommended_order}

Forecast:
{forecast_values}

Top SHAP Features:
{top_features.to_dict()}

Provide:

- demand outlook
- stock risks
- reorder reasoning
- inventory planning suggestions

Use concise business bullet points.
"""

        genai.configure(api_key=KEY_2)

        model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = model.generate_content(
            prompt
        )

        st.subheader(
            "🤖 AI Generated Insights"
        )

        st.info(response.text)

    except Exception as e:

        st.warning(
            f"Gemini insights failed: {e}"
        )