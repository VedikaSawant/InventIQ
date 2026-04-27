import hashlib
import logging

import chromadb
import uuid
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# =========================================================
# DEFAULTS
# =========================================================

DEFAULT_COLLECTION = "inventiq_knowledge"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "outputs/vector_store"

TOP_K_DEFAULT = 5


# =========================================================
# VECTOR STORE
# =========================================================

class VectorStore:

    def __init__(
        self,
        collection_name=DEFAULT_COLLECTION,
        embed_model=DEFAULT_EMBED_MODEL,
        persist_dir=DEFAULT_PERSIST_DIR,
    ):

        logger.info(f"Loading embedding model: {embed_model}")

        self.embedder = SentenceTransformer(embed_model)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # =====================================================
    # UPSERT
    # =====================================================

    def upsert(self, chunks):

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [sanitize_metadata(c["metadata"]) for c in chunks]

        ids = [str(uuid.uuid4()) for _ in texts]

        embeddings = self.embedder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    # =====================================================
    # QUERY
    # =====================================================

    def query(
        self,
        query_text,
        top_k=TOP_K_DEFAULT,
        filter_meta=None,
    ):

        embedding = self.embedder.encode(
            [query_text],
            convert_to_numpy=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            where=filter_meta,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(
                docs,
                metas,
                distances,
            )
        ]

    # =====================================================
    # ITEM QUERY
    # =====================================================

    def query_for_item(
        self,
        query_text,
        item_id,
        top_k=TOP_K_DEFAULT,
    ):

        results = self.query(
            query_text,
            top_k,
            {"item_id": item_id}
        )

        if len(results) < 2:

            global_results = self.query(
                query_text,
                top_k,
                {"item_id": "global"}
            )

            seen = {r["text"] for r in results}

            extra = [
                r for r in global_results
                if r["text"] not in seen
            ]

            results += extra[:top_k - len(results)]

        return results[:top_k]

    # =====================================================
    # UTILITIES
    # =====================================================

    def count(self):

        return self.collection.count()

    def reset(self):

        self.client.delete_collection(
            self.collection.name
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )


# =========================================================
# HELPERS
# =========================================================

def chunk_id(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:32]


def sanitize_metadata(meta):

    clean = {}

    for k, v in meta.items():

        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)

    return clean