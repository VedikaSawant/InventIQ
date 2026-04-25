"""
src/api/routers/chat.py
-------------------------
POST /chat/
    → Accepts a natural language question from the store manager
    → Retrieves relevant chunks from ChromaDB (RAG)
    → Streams the LLM response token-by-token via Server-Sent Events (SSE)

This is the conversational interface layer that makes everything explainable.
The manager asks plain English questions; the system answers using:
    - Retrieved SHAP explanation text (from ingestion.py)
    - Retrieved domain knowledge (reorder theory, safety stock formulas)
    - Current inventory state (inj  ected into the prompt as context)

LLM contract
-------------
We call the Google Gemini API.
The system prompt grounds the LLM in inventory management vocabulary.

SSE format
-----------
Each streamed event is a JSON line:
    data: {"token": "...", "done": false, "sources": [], "session_id": null}
    ...
    data: {"token": null,  "done": true,  "sources": ["chunk1...", "chunk2..."], "session_id": null}
"""

import json
import logging
import os
import google.generativeai as genai
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.models import ChatRequest, ChatEventPayload

logger = logging.getLogger(__name__)
router = APIRouter()

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

LLM_MODEL = "gemini-1.5-flash"
TOP_K_RETRIEVAL   = 5
MAX_TOKENS        = 512


SYSTEM_PROMPT = """You are InventIQ Assistant, an AI advisor for a store manager using
an AI-driven inventory optimization system.

You answer questions about:
- Why the system recommended a specific order quantity
- What the demand forecast means for stock planning
- Inventory management concepts (EOQ, safety stock, reorder points)
- How to interpret SHAP feature importance values

Rules:
- Be concise and practical. The user is a store manager, not a data scientist.
- Always ground your answers in the retrieved context provided.
- If the context does not contain enough information, say so clearly.
- When explaining SHAP values, use plain language: "the current stock level strongly
  pushed the system toward ordering more" rather than "SHAP=+0.42 for current_stock".
- Never make up specific numbers. Use only numbers from the retrieved context.
"""


@router.post("/")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    """
    Stream an LLM response grounded in RAG-retrieved inventory knowledge.

    The response is a Server-Sent Events stream. Each chunk is a JSON payload.
    The final event contains `done: true` and the list of source excerpts used.

    Example client consumption (JavaScript):
        const es = new EventSource('/chat/');
        es.onmessage = (e) => {
            const payload = JSON.parse(e.data);
            if (payload.done) { showSources(payload.sources); es.close(); }
            else              { appendToken(payload.token); }
        };
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY not set. Configure it in your .env file.",
        )

    ctx = request.app.state.ctx

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    if body.item_id:
        retrieved = ctx.vector_store.query_for_item(
            query_text = body.message,
            item_id    = body.item_id,
            top_k      = TOP_K_RETRIEVAL,
        )
    else:
        retrieved = ctx.vector_store.query(
            query_text = body.message,
            top_k      = TOP_K_RETRIEVAL,
        )

    # ── Build grounded prompt ─────────────────────────────────────────────────
    context_block = _format_context(retrieved)
    user_message  = (
        f"Context from the inventory system:\n{context_block}\n\n"
        f"Manager question: {body.message}"
    )

    source_excerpts = [
        r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        for r in retrieved
    ]

    return StreamingResponse(
        _stream_llm(user_message, source_excerpts, body.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering for SSE
        },
    )


# ── SSE generator ─────────────────────────────────────────────────────────────

async def _stream_llm(
    user_message: str,
    source_excerpts: list[str],
    session_id: str | None,
):
    """
    Stream response using Google Gemini.
    """

    try:
        model = genai.GenerativeModel(LLM_MODEL)

        response = model.generate_content(
            SYSTEM_PROMPT + "\n\n" + user_message,
            stream=True
        )

        for chunk in response:
            if chunk.text:
                payload = ChatEventPayload(
                    token=chunk.text,
                    done=False,
                    session_id=session_id,
                )

                yield f"data: {payload.model_dump_json()}\n\n"

        final = ChatEventPayload(
            token=None,
            done=True,
            sources=source_excerpts,
            session_id=session_id,
        )

        yield f"data: {final.model_dump_json()}\n\n"

    except Exception as e:
        error_payload = {"error": str(e), "done": True}
        yield f"data: {json.dumps(error_payload)}\n\n"
        logger.error(f"Gemini API error: {e}")

# ── Context formatter ─────────────────────────────────────────────────────────

def _format_context(retrieved: list[dict]) -> str:
    """
    Format retrieved chunks into a clean numbered context block for the LLM prompt.
    Numbered so the LLM can cite sources naturally ("According to context [2]...").
    """
    if not retrieved:
        return "No relevant context found in the knowledge base."

    lines = []
    for i, chunk in enumerate(retrieved, start=1):
        src  = chunk["metadata"].get("source_type", "unknown")
        item = chunk["metadata"].get("item_id", "global")
        lines.append(f"[{i}] ({src} | item={item})\n{chunk['text']}")

    return "\n\n".join(lines)