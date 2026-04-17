"""RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Buduje hierarchiczny indeks: chunki (liście) → klastry → streszczenia → wyższe klastry.
Retrieval na dowolnym poziomie abstrakcji (detale LUB podsumowania).
"""

import json
import uuid
from dataclasses import dataclass, field

import numpy as np
import anthropic

from config.settings import get_claude_config
from src.embeddings.jina_embed import JinaDenseEmbedder

SUMMARIZE_CLUSTER_PROMPT = """Napisz zwięzłe streszczenie (3-5 zdań) poniższego zbioru fragmentów tekstu.
Zachowaj kluczowe fakty i konkrety. Pisz po polsku.

Fragmenty:
{texts}

Streszczenie:"""


@dataclass
class RAPTORNode:
    """Węzeł w drzewie RAPTOR."""

    node_id: str
    text: str
    level: int  # 0=liść (chunk), 1=klaster, 2=wyższy klaster...
    embedding: list[float] = field(default_factory=list, repr=False)
    children_ids: list[str] = field(default_factory=list)
    is_summary: bool = False


@dataclass
class RAPTORTree:
    """Pełne drzewo RAPTOR."""

    nodes: dict[str, RAPTORNode] = field(default_factory=dict)
    levels: int = 0

    def get_level(self, level: int) -> list[RAPTORNode]:
        return [n for n in self.nodes.values() if n.level == level]

    def all_nodes_flat(self) -> list[RAPTORNode]:
        """Collapsed tree — wszystkie węzły jako flat list."""
        return list(self.nodes.values())


class RAPTORBuilder:
    """Buduje drzewo RAPTOR z chunków."""

    def __init__(
        self,
        embedder: JinaDenseEmbedder | None = None,
        cluster_size: int = 5,
        max_levels: int = 3,
    ):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model
        self._embedder = embedder or JinaDenseEmbedder()
        self._cluster_size = cluster_size
        self._max_levels = max_levels

    def _summarize(self, texts: list[str]) -> str:
        """Wygeneruj streszczenie klastra."""
        combined = "\n---\n".join(t[:1000] for t in texts)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            messages=[{"role": "user", "content": SUMMARIZE_CLUSTER_PROMPT.format(texts=combined[:8000])}],
        )
        return response.content[0].text.strip()

    def _cluster_by_similarity(
        self, nodes: list[RAPTORNode], cluster_size: int
    ) -> list[list[RAPTORNode]]:
        """Proste klastrowanie na podstawie cosine similarity embeddingów."""
        if len(nodes) <= cluster_size:
            return [nodes]

        embeddings = np.array([n.embedding for n in nodes])
        # Normalizacja
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        normalized = embeddings / norms

        n_clusters = max(2, len(nodes) // cluster_size)
        # Prosty k-means (bez sklearn)
        clusters = self._simple_kmeans(normalized, n_clusters)

        result = []
        for cluster_ids in clusters:
            result.append([nodes[i] for i in cluster_ids])

        return result

    def _simple_kmeans(self, data: np.ndarray, k: int, max_iter: int = 20) -> list[list[int]]:
        """Prosty k-means bez zewnętrznych zależności."""
        n = len(data)
        # Losowe centroidy
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=k, replace=False)
        centroids = data[indices].copy()

        assignments = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Przypisz do najbliższego centroidu
            sims = data @ centroids.T
            new_assignments = np.argmax(sims, axis=1)

            if np.all(assignments == new_assignments):
                break
            assignments = new_assignments

            # Przelicz centroidy
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    centroids[j] = data[mask].mean(axis=0)
                    centroids[j] /= np.linalg.norm(centroids[j]) + 1e-10

        # Konwertuj na listy indeksów
        clusters = [[] for _ in range(k)]
        for i, c in enumerate(assignments):
            clusters[c].append(i)

        return [c for c in clusters if c]  # Usuń puste

    def build(self, texts: list[str], progress_callback=None) -> RAPTORTree:
        """Zbuduj drzewo RAPTOR z listy tekstów.

        Args:
            texts: Lista chunków/fragmentów.
            progress_callback: Opcjonalny callback(progress, text).

        Returns:
            RAPTORTree z hierarchią.
        """
        tree = RAPTORTree()

        # Level 0: liście (chunki)
        if progress_callback:
            progress_callback(0.1, "Embeddowanie chunków (liście)...")

        embeddings = self._embedder.embed_documents(texts)

        leaf_nodes = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            node = RAPTORNode(
                node_id=f"L0_{i}",
                text=text,
                level=0,
                embedding=emb,
                is_summary=False,
            )
            tree.nodes[node.node_id] = node
            leaf_nodes.append(node)

        # Buduj wyższe poziomy
        current_nodes = leaf_nodes
        level = 1

        while level <= self._max_levels and len(current_nodes) > 1:
            if progress_callback:
                progress_callback(
                    0.1 + 0.8 * level / self._max_levels,
                    f"Budowanie poziomu {level} ({len(current_nodes)} węzłów → klastry)...",
                )

            clusters = self._cluster_by_similarity(current_nodes, self._cluster_size)

            if len(clusters) <= 1 and level > 1:
                break

            next_level_nodes = []
            for j, cluster in enumerate(clusters):
                # Streszczenie klastra
                cluster_texts = [n.text for n in cluster]
                summary = self._summarize(cluster_texts)

                # Embed streszczenia
                summary_emb = self._embedder.embed_documents([summary])[0]

                summary_node = RAPTORNode(
                    node_id=f"L{level}_{j}",
                    text=summary,
                    level=level,
                    embedding=summary_emb,
                    children_ids=[n.node_id for n in cluster],
                    is_summary=True,
                )
                tree.nodes[summary_node.node_id] = summary_node
                next_level_nodes.append(summary_node)

            current_nodes = next_level_nodes
            level += 1

        tree.levels = level - 1

        if progress_callback:
            progress_callback(1.0, "Drzewo RAPTOR zbudowane!")

        return tree

    def search(
        self,
        tree: RAPTORTree,
        query: str,
        top_k: int = 5,
        mode: str = "collapsed",
    ) -> list[RAPTORNode]:
        """Szukaj w drzewie RAPTOR.

        Args:
            tree: Drzewo RAPTOR.
            query: Zapytanie.
            top_k: Ile wyników.
            mode: 'collapsed' (flat search po wszystkich) lub 'tree' (traversal od roota).

        Returns:
            Lista najbardziej pasujących node'ów.
        """
        query_emb = np.array(self._embedder.embed_query(query))

        if mode == "collapsed":
            # Flat search po WSZYSTKICH node'ach (liście + summaries)
            all_nodes = tree.all_nodes_flat()
        else:
            # Tylko liście + top-level summaries
            all_nodes = tree.get_level(0) + tree.get_level(tree.levels)

        # Cosine similarity
        scored = []
        for node in all_nodes:
            node_emb = np.array(node.embedding)
            sim = float(np.dot(query_emb, node_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(node_emb) + 1e-10
            ))
            scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def close(self):
        self._client.close()
        self._embedder.close()
