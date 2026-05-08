import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

# =========================================================
# CONSTANTS
# =========================================================

SEQ_LEN = 28
HORIZON = 7
N_FEATURES = 17

ACTION_LEVELS = 5
MAX_ORDER = 20
ORDER_STEP = MAX_ORDER // (ACTION_LEVELS - 1)

STATE_DIM = 1 + HORIZON + 1 + 1

MAX_ORDER_GAP = 14
MAX_STOCKOUT_STREAK = 7


# =========================================================
# ENVIRONMENT
# =========================================================

class InventoryEnv(gym.Env):

    def __init__(
        self,
        demand_series,
        feature_matrix,
        forecaster,
        scaler,
        target_scaler,
        initial_stock=5,
        holding_cost=0.05,
        stockout_penalty=25.0,
        max_stock=150,
    ):

        super().__init__()
        self.unmet_values = []

        assert len(demand_series) == len(feature_matrix)
        assert len(demand_series) > SEQ_LEN + HORIZON

        self.demand = demand_series.astype(np.float32)
        self.features = feature_matrix.astype(np.float32)

        self.forecaster = forecaster
        self.scaler = scaler
        self.target_scaler = target_scaler

        self.initial_stock = initial_stock
        self.holding_cost = holding_cost
        self.stockout_pen = stockout_penalty
        self.max_stock = max_stock

        self.action_space = spaces.Discrete(ACTION_LEVELS)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        self.reset()

    # =====================================================
    # RESET
    # =====================================================

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self.unmet_values = []

        self.t = SEQ_LEN
        self.stock = float(self.initial_stock)

        self.days_since_order = 0
        self.stockout_streak = 0

        obs = self._get_obs()

        return obs, {"stock": self.stock}

    # =====================================================
    # STEP
    # =====================================================

    def step(self, action):

        assert self.action_space.contains(action)

        order_qty = action * ORDER_STEP

        self._apply_order(order_qty)

        reward, demand_today, unmet = self._fulfill_demand()
        self.unmet_values.append(unmet)

        self._update_stockout(unmet)

        self.t += 1

        terminated = self.t >= len(self.demand) - HORIZON

        if terminated and len(self.unmet_values) > 0:

            arr = np.array(self.unmet_values)

            print("\n===== UNMET STATISTICS =====")

            print("Total steps:", len(arr))

            print(
                "Unmet > 0 count:",
                np.sum(arr > 0)
            )

            print(
                "Percent unmet:",
                round(100 * np.mean(arr > 0), 2),
                "%"
            )

            print(
                "Max unmet:",
                round(np.max(arr), 2)
            )

            print(
                "Avg unmet:",
                round(np.mean(arr), 2)
            )

            print(
                "Min unmet:",
                round(np.min(arr), 2)
            )

            print("============================\n")

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(STATE_DIM, dtype=np.float32)
        )

        info = {
            "order_qty": order_qty,
            "demand_today": demand_today,
            "unmet_demand": unmet,
            "stock": self.stock,
            "reward": reward,
        }

        # print(
        #     "order_qty:", order_qty,
        #     "days_since_order:", self.days_since_order,
        #     "stockout_streak:", self.stockout_streak
        # )

        # print("Stock:", self.stock)
        # print("Demand:", demand_today)
        # print("Unmet:", unmet)
        # print("Stockout streak:", self.stockout_streak)
        # print("Days since order:", self.days_since_order)

        return obs, reward, terminated, False, info

    # =====================================================
    # INTERNAL LOGIC
    # =====================================================

    def _apply_order(self, order_qty):

        if order_qty > 0:
            self.days_since_order = 0
        else:
            self.days_since_order += 1

        self.stock = min(
            self.stock + order_qty,
            self.max_stock
        )

    def _fulfill_demand(self):

        base_demand = float(self.demand[self.t])

        noise = np.random.normal(
            0,
            max(1, 0.1 * base_demand)
        )

        demand_today = max(
            (base_demand + noise),
            0.0
        )

        units_sold = min(self.stock, demand_today)

        unmet = max(demand_today - self.stock, 0.0)

        self.stock -= units_sold

        holding_cost = self.holding_cost * self.stock

        stockout_cost = self.stockout_pen * unmet

        reward = -(holding_cost + stockout_cost)

        overstock_penalty = 0.5 * max(
            self.stock - 0.8 * self.max_stock,
            0
        )

        reward -= overstock_penalty

        return reward, demand_today, unmet

    def _update_stockout(self, unmet):

        if unmet > 0:
            self.stockout_streak += 1
        else:
            self.stockout_streak = 0

    # =====================================================
    # OBSERVATION
    # =====================================================

    def _get_obs(self):

        forecast = self._forecast()

        stock_norm = self.stock / self.max_stock

        forecast_norm = forecast / (
            self.max_stock + 1e-8
        )

        days_norm = (
            self.days_since_order
            / MAX_ORDER_GAP
        )

        streak_norm = (
            self.stockout_streak
            / MAX_STOCKOUT_STREAK
        )

        obs = np.concatenate([
            [stock_norm],
            forecast_norm,
            [days_norm],
            [streak_norm],
        ]).astype(np.float32)

        return obs

    # =====================================================
    # FORECAST
    # =====================================================

    def _forecast(self):

        raw_window = self.features[
            self.t - SEQ_LEN : self.t
        ]

        return self._run_forecast(raw_window)

    def _run_forecast(self, raw_window):

        scaled = self.scaler.transform(raw_window)

        tensor = torch.from_numpy(
            scaled.astype(np.float32)
        )

        with torch.no_grad():

            scaled_preds = self.forecaster(
                tensor.unsqueeze(0)
            ).squeeze(0).numpy()

        raw = np.expm1(
            self.target_scaler.inverse_transform(
                scaled_preds.reshape(-1, 1)
            ).flatten()
        )

        return np.maximum(raw, 0.0)

    # =====================================================
    # RENDER
    # =====================================================

    def render(self):

        print(
            f"Day {self.t} | "
            f"Stock: {self.stock:.1f} | "
            f"Stockout streak: {self.stockout_streak}"
        )