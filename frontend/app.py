# frontend/app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

from pathlib import Path
import matplotlib.pyplot as plt


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

        "🧠 Explainability",

        "💬 Chatbot"

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

    st.title("🏪 Store Dashboard")

    if selected_item:

        item_df = df[

            df[ITEM_COL] == selected_item

        ]

    else:

        item_df = df.copy()


    # Try common demand column names

    DEMAND_COL = None

    for c in [

        "sales",

        "demand",

        "units_sold"

    ]:

        if c in item_df.columns:

            DEMAND_COL = c

            break


    if DEMAND_COL is None:

        st.warning(

            "Demand column not found."

        )

    else:

        recent_sales = item_df[DEMAND_COL].tail(30)


        # Fake stock estimate from demand

        current_stock = int(

            max(

                50 - recent_sales.iloc[-1],

                5

            )

        )


        predicted_demand = int(

            recent_sales.mean()

        )


        recommended_order = max(

            predicted_demand - current_stock,

            0

        )


        # KPI Metrics

        col1, col2, col3 = st.columns(3)


        col1.metric(

            "Current Stock",

            current_stock

        )


        col2.metric(

            "Predicted Demand",

            predicted_demand

        )


        col3.metric(

            "Recommended Order",

            recommended_order

        )


        # Sales Trend

        st.subheader(

            "Recent Demand Trend"

        )


        fig, ax = plt.subplots()

        ax.plot(

            recent_sales.values

        )

        ax.set_title(

            "Last 30 Days Demand"

        )


        st.pyplot(fig)



# =========================================================
# FORECAST PAGE
# =========================================================

elif page == "📦 Forecast":

    st.title("📦 Forecast View")

    # Example forecast display

    st.info(

        "Forecast data should come from model predictions."

    )


    # Temporary visualization

    forecast_days = np.arange(1, 8)

    forecast_values = np.random.randint(

        20,

        100,

        size=7

    )


    fig, ax = plt.subplots()

    ax.plot(

        forecast_days,

        forecast_values

    )


    ax.set_title(

        "7-Day Forecast"

    )

    ax.set_xlabel("Day")

    ax.set_ylabel("Demand")


    st.pyplot(fig)



# =========================================================
# DECISIONS PAGE
# =========================================================

elif page == "🤖 Decisions":

    st.title("🤖 AI Decision Output")


    st.subheader(

        "Recommended Order"

    )


    current_stock = np.random.randint(

        10,

        80

    )


    forecast_total = np.random.randint(

        50,

        140

    )


    recommended_order = max(

        forecast_total - current_stock,

        0

    )


    st.metric(

        "Order Quantity",

        recommended_order

    )


    st.success(

        "Low stock detected — ordering recommended."

    )



# =========================================================
# EXPLAINABILITY
# =========================================================

elif page == "🧠 Explainability":

    st.title("🧠 SHAP Explainability")


    if not SHAP_DIR.exists():

        st.warning(

            "No SHAP directory found."

        )


    else:

        images = list(

            SHAP_DIR.glob("*.png")

        )


        if not images:

            st.warning(

                "No SHAP plots available."

            )

        else:

            selected_image = st.selectbox(

                "Select SHAP Plot",

                images

            )


            st.image(

                selected_image,

                caption="Feature Importance"

            )



# =========================================================
# CHATBOT
# =========================================================

elif page == "💬 Chatbot":

    st.title("💬 Inventory Assistant")


    if "messages" not in st.session_state:

        st.session_state.messages = []


    user_input = st.text_input(

        "Ask about inventory decisions:"

    )


    if user_input:


        payload = {

            "message": user_input,

            "item_id": selected_item,

            "session_id": None

        }


        try:


            response = requests.post(

                API_URL,

                json=payload,

                stream=True

            )


            full_response = ""


            for line in response.iter_lines():


                if line:


                    decoded = line.decode()


                    if decoded.startswith("data:"):


                        data = decoded.replace(

                            "data:", ""

                        ).strip()


                        try:


                            json_data = json.loads(

                                data

                            )


                            token = json_data.get(

                                "token"

                            )


                            if token:

                                full_response += token


                        except:

                            pass


            st.session_state.messages.append(

                ("user", user_input)

            )


            st.session_state.messages.append(

                ("bot", full_response)

            )


        except Exception as e:


            st.error(

                f"Chat error: {e}"

            )


    for role, msg in st.session_state.messages:


        if role == "user":

            st.write(f"🧑 {msg}")


        else:

            st.write(f"🤖 {msg}")