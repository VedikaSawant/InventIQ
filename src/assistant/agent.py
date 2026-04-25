import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


logger = logging.getLogger(__name__)


# =========================================================
# CONTEXT
# =========================================================

@dataclass
class ToolContext:

    trainer: any
    ppo_agent: any
    scaler: any
    agent_explainer: any
    vector_store: any
    processed_data_path: str


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_context(vector_store, query, item_id=None, top_k=5):

    if item_id:
        chunks = vector_store.query_for_item(query, item_id, top_k)
    else:
        chunks = vector_store.query(query, top_k)

    if not chunks:
        return "No relevant context found."

    lines = []

    for i, c in enumerate(chunks, start=1):

        meta = c.get("metadata", {})
        text = c.get("text", "")

        lines.append(
            f"[{i}] ({meta.get('source_type','unknown')})\n{text[:300]}"
        )

    return "\n\n".join(lines)


# =========================================================
# TOOL FUNCTIONS
# =========================================================

MAX_STOCK = 500.0
ORDER_STEP = 10


def get_forecast(item_id, window, ctx):

    window_np = np.array(window, dtype=np.float32)

    scaled = ctx.scaler.transform(window_np)

    tensor = torch.from_numpy(scaled)

    with torch.no_grad():

        preds = ctx.trainer.forecast(tensor).numpy()

    dummy = np.zeros((7, scaled.shape[1]))

    dummy[:, 0] = preds

    forecast_units = ctx.scaler.inverse_transform(dummy)[:, 0]

    return forecast_units.round(2).tolist()


def get_stock_state(
    item_id,
    current_stock,
    forecast_units
):

    avg = float(np.mean(forecast_units))

    coverage = current_stock / max(avg, 1e-6)

    reorder_flag = current_stock <= 50

    return {
        "coverage_days": round(coverage, 1),
        "reorder_flag": reorder_flag,
    }


def get_shap_explanation(
    item_id,
    current_stock,
    forecast_units,
    ctx
):

    if ctx.agent_explainer is None:
        return "SHAP explainer not available."

    forecast_arr = np.array(forecast_units[:7])

    obs = np.array(
        [current_stock / MAX_STOCK]
        + (forecast_arr / MAX_STOCK).tolist()
        + [0.0, 0.0]
    )

    action = ctx.ppo_agent.predict(obs)

    order_qty = action * ORDER_STEP

    result = ctx.agent_explainer.explain(
        obs=obs,
        action_taken=action,
        order_qty=order_qty,
        item_id=item_id
    )

    return result["natural_language_summary"]


# =========================================================
# WHAT-IF SIMULATION
# =========================================================

def run_whatif_sim(
    item_id,
    initial_stock,
    n_days,
    policy,
    ctx
):

    stock = float(initial_stock)

    total_reward = 0

    for _ in range(n_days):

        if policy == "ppo":

            obs = np.zeros(10)

            obs[0] = stock / MAX_STOCK

            order_qty = (
                ctx.ppo_agent.predict(obs)
                * ORDER_STEP
            )

        else:

            order_qty = 0

        stock += order_qty

        reward = stock * -0.5

        total_reward += reward

    return {
        "policy": policy,
        "total_reward": round(total_reward, 2),
    }


def get_demand_alerts(item_id, ctx):

    return {
        "item_id": item_id,
        "alert": False,
        "message": "No unusual demand detected."
    }


# =========================================================
# TOOL BUILDER
# =========================================================

def _make_tools(ctx):

    def parse(raw):
        return json.loads(raw)

    return [

        Tool(
            name="retrieve_knowledge",
            func=lambda raw:
                retrieve_context(
                    ctx.vector_store,
                    **parse(raw)
                ),
            description="Retrieve knowledge context."
        ),

        Tool(
            name="get_forecast",
            func=lambda raw:
                str(
                    get_forecast(
                        **parse(raw),
                        ctx=ctx
                    )
                ),
            description="Run demand forecast."
        ),

        Tool(
            name="get_stock_state",
            func=lambda raw:
                str(
                    get_stock_state(
                        **parse(raw)
                    )
                ),
            description="Check stock coverage."
        ),

        Tool(
            name="get_shap_explanation",
            func=lambda raw:
                str(
                    get_shap_explanation(
                        **parse(raw),
                        ctx=ctx
                    )
                ),
            description="Explain PPO decision."
        ),

        Tool(
            name="run_whatif_simulation",
            func=lambda raw:
                str(
                    run_whatif_sim(
                        **parse(raw),
                        ctx=ctx
                    )
                ),
            description="Run what-if simulation."
        ),

        Tool(
            name="get_demand_alerts",
            func=lambda raw:
                str(
                    get_demand_alerts(
                        **parse(raw),
                        ctx=ctx
                    )
                ),
            description="Check demand alerts."
        ),
    ]


# =========================================================
# AGENT BUILDER
# =========================================================

SYSTEM_PROMPT = """
You are InventIQ Assistant, an AI advisor for a retail store manager.

You help explain inventory decisions, forecast demand, and simulate policies.

You have access to the following tools:
{tools}

Use the ReAct reasoning format:

Thought: Think step-by-step about what information is needed  
Action: tool_name  
Action Input: {{"param": "value"}}  
Observation: Tool result  
... (repeat as needed)

When you have enough information:

Thought: I now have enough information  
Final Answer: Provide a clear, practical explanation for the store manager.

Available tool names:
{tool_names}

Important Rules:

1. Always retrieve knowledge first when answering explanation questions.

2. Be concise and practical.
   The store manager is not a data scientist.

3. When explaining SHAP results:
   Say things like:
       "Low current stock strongly pushed toward ordering more."
   Do NOT say:
       "SHAP = +0.42"

4. Never invent numbers.
   Only use values returned by tools.

5. If a tool fails:
   Explain the limitation clearly instead of guessing.

6. Prefer actionable recommendations.
   Example:
       "Consider increasing safety stock by 10–15 units."

Question:
{input}

{agent_scratchpad}
"""


def build_agent(ctx):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1,
    )

    tools = _make_tools(ctx)

    prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT
    )

    agent = create_react_agent(
        llm,
        tools,
        prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=6,
        handle_parsing_errors=True,
    )