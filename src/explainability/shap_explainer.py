import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = Path("outputs/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_FEATURE_NAMES = [
    "sales", "sell_price", "wday", "month",
    "is_event", "is_snap",
    "lag_7", "lag_28", "rolling_7",
]

AGENT_FEATURE_NAMES = [
    "current_stock",
    "forecast_d1", "forecast_d2",
    "forecast_d3", "forecast_d4",
    "forecast_d5", "forecast_d6",
    "forecast_d7",
    "days_since_order",
    "stockout_streak",
]


# =========================================================
# GENERIC PLOTTING
# =========================================================

def _plot_bar(importances, title, filename):

    names = list(importances.keys())
    values = list(importances.values())

    order = np.argsort(values)[::-1]

    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(names, values)

    ax.set_title(title)

    ax.invert_yaxis()

    path = OUTPUT_DIR / filename

    fig.savefig(path)

    plt.close(fig)

    return str(path)


def _plot_waterfall(values, title, filename):

    fig, ax = plt.subplots(figsize=(9, 5))

    # Sort features by importance
    order = np.argsort(np.abs(values))[::-1]

    # Keep only valid indexes
    safe_order = [
        i for i in order
        if i < len(AGENT_FEATURE_NAMES)
    ]

    # Apply safe ordering
    sorted_values = values[safe_order]

    names = [
        AGENT_FEATURE_NAMES[i]
        for i in safe_order
    ]

    colors = [
        "red" if v > 0 else "blue"
        for v in sorted_values
    ]

    ax.barh(names, sorted_values, color=colors)

    ax.axvline(0)

    ax.set_title(title)

    ax.invert_yaxis()

    path = OUTPUT_DIR / filename

    fig.savefig(path)

    plt.close(fig)

    return str(path)


# =========================================================
# FORECAST EXPLAINER
# =========================================================

class ForecastExplainer:

    def __init__(self, model, background):

        self.model = model

        self.explainer = shap.DeepExplainer(
            model,
            background
        )

    def explain(
        self,
        window,
        item_id="unknown",
        horizon_day=0
    ):

        if window.dim() == 2:
            window = window.unsqueeze(0)

        shap_values = self.explainer.shap_values(window)

        sv = shap_values[horizon_day][0]

        importance = np.abs(sv).mean(axis=0)

        feature_importances = {
            name: float(importance[i])
            for i, name in enumerate(
                FORECAST_FEATURE_NAMES
            )
        }

        plot_path = _plot_bar(
            feature_importances,
            title=f"{item_id} day+{horizon_day+1}",
            filename=f"forecast_{item_id}.png"
        )

        return {
            "item_id": item_id,
            "feature_importances": feature_importances,
            "plot_path": plot_path,
        }


# =========================================================
# AGENT EXPLAINER
# =========================================================

class AgentExplainer:

    def __init__(self, predict_fn, background):

        self.explainer = shap.KernelExplainer(
            predict_fn,
            shap.kmeans(background, 20)
        )

    def explain(
        self,
        obs,
        action_taken,
        order_qty,
        item_id="unknown",
        step=0
    ):

        obs_2d = obs.reshape(1, -1)

        shap_values = self.explainer.shap_values(
            obs_2d,
            nsamples=100
        )

        print("\n========== SHAP DEBUG ==========")
        print("Observation shape:", obs.shape)
        print("Observation values:", obs)

        print("Raw SHAP values:", shap_values)

        sv = shap_values[0][action_taken]
        print("Selected SHAP values:", sv)
        print("Max SHAP abs:", np.max(np.abs(sv)))

        print("Flattened SHAP length:", len(sv))
        print("Feature names length:", len(AGENT_FEATURE_NAMES))

        print("Action taken:", action_taken)
        print("Order qty:", order_qty)

        print("================================\n")

        sv = np.array(shap_values).reshape(-1)

        feature_importances = {
            name: float(sv[i])
            for i, name in enumerate(
                AGENT_FEATURE_NAMES
            )
        }

        plot_path = _plot_waterfall(
            sv,
            title=f"{item_id} step {step}",
            filename=f"agent_{item_id}.png"
        )

        summary = self._build_summary(
            feature_importances,
            order_qty,
            item_id
        )

        return {
            "item_id": item_id,
            "action_taken": action_taken,
            "order_qty": order_qty,
            "feature_importances": feature_importances,
            "plot_path": plot_path,
            "natural_language_summary": summary,
        }

    # =====================================================
    # NATURAL LANGUAGE SUMMARY
    # =====================================================

    def _build_summary(
        self,
        importances,
        order_qty,
        item_id
    ):

        sorted_features = sorted(
            importances.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True
        )

        top3 = sorted_features[:3]

        lines = []

        for feat, val in top3:

            direction = (
                "increased"
                if val > 0
                else "decreased"
            )

            lines.append(
                f"{feat} {direction} the order quantity"
            )

        return (
            f"Item {item_id}: ordered {order_qty} units.\n"
            + "\n".join(lines)
        )


# =========================================================
# BUILDERS
# =========================================================

def build_forecast_explainer(
    model,
    train_loader,
    n_background=100
):

    batches = []

    count = 0

    for x, _ in train_loader:

        batches.append(x)

        count += len(x)

        if count >= n_background:
            break

    background = torch.cat(batches, dim=0)[:n_background]

    return ForecastExplainer(
        model,
        background
    )


def build_agent_explainer(
    agent,
    rollout_states
):

    def predict_fn(obs_array):

        t = torch.from_numpy(
            obs_array.astype(np.float32)
        )

        with torch.no_grad():

            logits, _ = agent.policy(t)

            probs = torch.softmax(
                logits,
                dim=-1
            )

        return probs.numpy()

    return AgentExplainer(
        predict_fn,
        rollout_states
    )