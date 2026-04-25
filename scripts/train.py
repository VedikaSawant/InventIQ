# scripts/train.py

import argparse
import logging
import pickle
import yaml
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger(__name__)


def main(args):

    # -----------------------------------------------------
    # Load config
    # -----------------------------------------------------

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # =====================================================
    # STEP 1 — RUN DATA PIPELINE
    # =====================================================

    if not args.skip_data:

        log.info("Running data pipeline...")

        from src.data.data_loader import run_data_pipeline

        df = run_data_pipeline()

        log.info(f"Pipeline complete. Shape: {df.shape}")


    # =====================================================
    # STEP 2 — TRAIN LSTM
    # =====================================================

    if not args.skip_lstm:

        log.info("Training LSTM...")

        from src.data.data_loader import get_dataloaders
        from src.forecasting.forecasting import (
            DemandForecastingSystem,
            DemandLSTM
        )

        data_cfg = cfg["data"]
        model_cfg = cfg.get("lstm", {})

        train_loader, val_loader, _, scaler = get_dataloaders(
            batch_size=args.batch_size
        )

        system = DemandForecastingSystem()

        system.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs
        )

        # Save scaler

        scaler_path = "outputs/models/scaler.pkl"

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        log.info("LSTM training complete.")


    # =====================================================
    # STEP 3 — TRAIN PPO
    # =====================================================

    if not args.skip_ppo:

        log.info("Training PPO agent...")

        from src.environment.inventory_env import InventoryEnv
        from src.agent.rl_agent import PPOAgent
        from src.forecasting.forecasting import (
            DemandForecastingSystem,
            DemandLSTM
        )
        from src.data.data_loader import get_dataloaders

        # Load trained LSTM

        system = DemandForecastingSystem()

        system.load()

        forecaster = system.freeze_for_rl()

        # Load scaler

        with open("outputs/models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        train_loader, _, _, _ = get_dataloaders(
            batch_size=1
        )

        dataset = train_loader.dataset

        demand_series = dataset.y[:, 0]

        feature_matrix = dataset.X[:, -1, :]

        env = InventoryEnv(
            demand_series=demand_series,
            feature_matrix=feature_matrix,
            forecaster=forecaster,
            scaler=scaler,
        )

        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
        )

        history = agent.train(
            env,
            total_timesteps=args.timesteps,
        )

        agent.save("ppo_inventory.pt")

        # Collect rollout states

        rollout_states = _collect_rollout_states(
            agent,
            env
        )

        np.save(
            "outputs/models/rollout_states.npy",
            rollout_states
        )

        log.info("PPO training complete.")


def _collect_rollout_states(agent, env, n=200):

    states = []

    obs, _ = env.reset()

    while len(states) < n:

        states.append(obs.copy())

        action = agent.predict(obs)

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:

            obs, _ = env.reset()

    return np.array(states[:n])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config.yaml")

    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--timesteps", type=int, default=200_000)

    parser.add_argument("--skip_data", action="store_true")

    parser.add_argument("--skip_lstm", action="store_true")

    parser.add_argument("--skip_ppo", action="store_true")

    main(parser.parse_args())