"""Benchmarki modeli embeddingowych na polskich treściach.

Porównuje modele pod kątem hit@k i MRR na zestawie testowym.
In-memory cosine similarity (numpy), bez dodatkowych kolekcji Qdrant.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

AVAILABLE_MODELS = {
    "jina-v5": {
        "type": "jina",
        "name": "jina-embeddings-v5-text-small",
        "description": "Jina v5 text-small (API, 1024d)",
    },
    "e5-large": {
        "type": "local",
        "name": "intfloat/multilingual-e5-large",
        "description": "multilingual-e5-large (lokalne, 1024d)",
    },
    "bge-m3": {
        "type": "local",
        "name": "BAAI/bge-m3",
        "description": "BGE-M3 (lokalne, 1024d)",
    },
    "openai": {
        "type": "openai",
        "name": "text-embedding-3-small",
        "description": "OpenAI text-embedding-3-small (API, 1536d)",
    },
}


@dataclass
class BenchmarkResult:
    """Wynik benchmarku dla jednego modelu."""

    model_key: str
    model_name: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr: float
    avg_embed_time_ms: float
    embedding_dim: int


def _cosine_similarity_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Oblicz macierz cosine similarity."""
    # Normalizacja
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    d_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-10)
    return q_norm @ d_norm.T


def _get_embedder(model_key: str):
    """Zwróć instancję embeddiera na podstawie klucza modelu."""
    info = AVAILABLE_MODELS[model_key]

    if info["type"] == "jina":
        from src.embeddings.jina_embed import JinaDenseEmbedder
        return JinaDenseEmbedder()
    elif info["type"] == "local":
        from src.benchmarks.local_embedder import LocalSentenceTransformerEmbedder
        return LocalSentenceTransformerEmbedder(info["name"])
    elif info["type"] == "openai":
        from src.benchmarks.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()
    else:
        raise ValueError(f"Nieznany typ modelu: {info['type']}")


def run_benchmark(
    model_keys: list[str],
    test_questions: list[dict],
    chunks: list[dict],
    progress_callback=None,
) -> list[BenchmarkResult]:
    """Uruchom benchmark na wybranych modelach.

    Args:
        model_keys: Lista kluczy modeli z AVAILABLE_MODELS.
        test_questions: Lista dict z kluczami: question, source_article.
        chunks: Lista dict z kluczami: text, slug.
        progress_callback: Opcjonalny callback(progress: float, text: str).

    Returns:
        Lista BenchmarkResult per model.
    """
    results = []
    total_steps = len(model_keys)

    for step, model_key in enumerate(model_keys):
        info = AVAILABLE_MODELS[model_key]
        if progress_callback:
            progress_callback(step / total_steps, f"Benchmarking: {info['description']}...")

        embedder = _get_embedder(model_key)

        # Embed chunków
        chunk_texts = [c["text"] for c in chunks]
        t0 = time.time()
        chunk_vectors = embedder.embed_documents(chunk_texts)
        embed_time = (time.time() - t0) / len(chunk_texts) * 1000  # ms per chunk

        chunk_matrix = np.array(chunk_vectors)
        dim = chunk_matrix.shape[1]

        # Embed zapytań
        query_texts = [q["question"] for q in test_questions]
        query_vectors = [embedder.embed_query(q) for q in query_texts]
        query_matrix = np.array(query_vectors)

        # Cosine similarity
        sim_matrix = _cosine_similarity_matrix(query_matrix, chunk_matrix)

        # Metryki hit@k i MRR
        hits = {1: 0, 3: 0, 5: 0, 10: 0}
        mrr_sum = 0

        for i, q in enumerate(test_questions):
            # Sprawdź które chunki pochodzą z source_article
            source = q.get("source_article", "")
            # Wyciągnij slug z URL-a lub nazwy pliku
            if "/" in source:
                source_slug = source.rstrip("/").split("/")[-1]
            else:
                source_slug = source.replace(".md", "")

            relevant_indices = [
                j for j, c in enumerate(chunks)
                if source_slug and source_slug in c.get("slug", "")
            ]

            if not relevant_indices:
                continue

            # Ranking
            sorted_indices = np.argsort(sim_matrix[i])[::-1]

            # Hit@k
            for k in hits:
                top_k = set(sorted_indices[:k].tolist())
                if top_k & set(relevant_indices):
                    hits[k] += 1

            # MRR
            for rank, idx in enumerate(sorted_indices, 1):
                if idx in relevant_indices:
                    mrr_sum += 1.0 / rank
                    break

        n_queries = len(test_questions)
        result = BenchmarkResult(
            model_key=model_key,
            model_name=info["description"],
            hit_at_1=hits[1] / n_queries if n_queries else 0,
            hit_at_3=hits[3] / n_queries if n_queries else 0,
            hit_at_5=hits[5] / n_queries if n_queries else 0,
            hit_at_10=hits[10] / n_queries if n_queries else 0,
            mrr=mrr_sum / n_queries if n_queries else 0,
            avg_embed_time_ms=round(embed_time, 1),
            embedding_dim=dim,
        )
        results.append(result)

        # Cleanup
        if hasattr(embedder, "close"):
            embedder.close()

    if progress_callback:
        progress_callback(1.0, "Benchmark zakończony!")

    return results
