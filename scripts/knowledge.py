"""
scripts/knowledge.py

Generate SHAP explanations for PPO agent decisions.

This script:
    1. Loads rollout states from PPO training
    2. Loads trained PPO agent
    3. Builds SHAP explainer
    4. Generates SHAP explanations
    5. Saves explanations locally

Usage:
    python scripts/knowledge.py
"""

import argparse
import logging
import pickle

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger(__name__)


def main(args):

    from src.agent.rl_agent import PPOAgent
    from src.explainability.shap_explainer import (
        build_agent_explainer
    )

    # =====================================================
    # LOAD CONFIG
    # =====================================================

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # =====================================================
    # LOAD PPO AGENT
    # =====================================================

    log.info("Loading PPO agent...")

    rollout_path = "outputs/models/rollout_states.npy"

    rollout_states = np.load(rollout_path)

    state_dim = rollout_states.shape[1]

    agent = PPOAgent(
        state_dim=state_dim,
        n_actions=11
    )

    agent.load("ppo_inventory.pt")

    agent.policy.eval()

    log.info(f"Loaded rollout states: {rollout_states.shape}")

    # =====================================================
    # BUILD SHAP EXPLAINER
    # =====================================================

    log.info("Building SHAP explainer...")

    explainer = build_agent_explainer(
        agent,
        rollout_states
    )

    # =====================================================
    # SELECT STATES TO EXPLAIN
    # =====================================================

    TARGET_SAMPLES = 150

    # Try to focus on interesting states
    interesting_indices = np.where(
        (rollout_states[:, -2] > 0) |
        (rollout_states[:, -1] > 0)
    )[0]

    if len(interesting_indices) > 0:

        sample_size = min(
            TARGET_SAMPLES,
            len(interesting_indices)
        )

        chosen_indices = np.random.choice(
            interesting_indices,
            size=sample_size,
            replace=False
        )

    else:

        sample_size = min(
            TARGET_SAMPLES,
            len(rollout_states)
        )

        chosen_indices = np.random.choice(
            len(rollout_states),
            size=sample_size,
            replace=False
        )

    # =====================================================
    # GENERATE SHAP EXPLANATIONS
    # =====================================================

    log.info("Generating SHAP explanations...")

    item_ids = cfg.get(
        "selected_items",
        ["UNKNOWN_ITEM"]
    )

    results = []

    for i, idx in enumerate(chosen_indices):

        obs = rollout_states[idx]

        action = agent.predict(obs)

        order_qty = action * 10

        result = explainer.explain(
            obs=obs,
            action_taken=action,
            order_qty=order_qty,
            item_id=item_ids[i % len(item_ids)],
            step=i,
        )

        results.append(result)

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    output_path = "outputs/shap_explanations.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    log.info(f"Generated {len(results)} SHAP explanations.")

    log.info(f"Saved explanations to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for PPO agent"
    )

    parser.add_argument(
        "--config",
        default="config.yaml"
    )

    main(parser.parse_args())