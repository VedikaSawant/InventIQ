import json
import re
from pathlib import Path
from datetime import datetime

MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 80


# =========================================================
# BASIC CHUNK BUILDER
# =========================================================

def make_chunk(text, metadata=None):

    if metadata is None:
        metadata = {}

    return {
        "text": text.strip(),
        "metadata": {
            "source_type": metadata.get("source_type", "unknown"),
            "item_id": metadata.get("item_id", "global"),
            "date": metadata.get(
                "date",
                datetime.utcnow().isoformat()
            ),
            **metadata
        }
    }


# =========================================================
# SHAP INGESTION
# =========================================================

def ingest_shap_results(shap_results):

    chunks = []

    for i, result in enumerate(shap_results):

        item_id = result.get("item_id", "unknown")

        source_type = (
            "agent_decision"
            if "order_qty" in result
            else "forecast"
        )

        base_meta = {
            "source_type": source_type,
            "item_id": item_id
        }

        # Natural language summary
        if "natural_language_summary" in result:

            text = (
                f"Decision step {i} for item {item_id}:\n"
                + result["natural_language_summary"]
            )

            chunks.append(
                make_chunk(
                    text,
                    {**base_meta,
                     "chunk_type": "decision_summary"}
                )
            )

        # Feature importance
        fi = result.get("feature_importances", {})

        if fi:

            ranked = sorted(
                fi.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True
            )

            feature_text = "; ".join(
                f"{k} ({abs(v):.3f})"
                for k, v in ranked[:6]
            )

            text = (
                f"For item {item_id}, important factors were: "
                + feature_text
            )

            chunks.append(
                make_chunk(
                    text,
                    {**base_meta,
                     "chunk_type": "feature_ranking"}
                )
            )

    return chunks


# =========================================================
# DOMAIN DOC INGESTION
# =========================================================

def ingest_domain_docs(docs_dir):

    docs_path = Path(docs_dir)

    if not docs_path.exists():
        return []

    chunks = []

    files = (
        list(docs_path.glob("*.txt"))
        + list(docs_path.glob("*.md"))
    )

    for file in files:

        text = file.read_text(encoding="utf-8")

        file_chunks = chunk_text(
            text,
            source_file=file.name
        )

        chunks.extend(file_chunks)

    return chunks


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(text, source_file):

    paragraphs = re.split(r"\n\n+", text)

    chunks = []

    buffer = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        if len(buffer) + len(para) < MAX_CHUNK_CHARS:

            buffer += "\n\n" + para

        else:

            if buffer:

                chunks.append(
                    make_chunk(
                        buffer,
                        {
                            "source_type": "domain_knowledge",
                            "source_file": source_file
                        }
                    )
                )

            buffer = para[-OVERLAP_CHARS:]

    if buffer:

        chunks.append(
            make_chunk(
                buffer,
                {
                    "source_type": "domain_knowledge",
                    "source_file": source_file
                }
            )
        )

    return chunks


# =========================================================
# MASTER BUILDER
# =========================================================

def build_knowledge_chunks(
    shap_results=None,
    docs_dir="src/knowledge/docs"
):

    chunks = []

    chunks.extend(
        ingest_domain_docs(docs_dir)
    )

    if shap_results:

        chunks.extend(
            ingest_shap_results(
                shap_results
            )
        )

    return chunks


# =========================================================
# DEBUG SAVE
# =========================================================

def save_chunks_to_jsonl(
    chunks,
    out_path
):

    Path(out_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(out_path, "w") as f:

        for chunk in chunks:

            f.write(
                json.dumps(chunk)
                + "\n"
            )