"""Moduł zarządzania kolekcjami Qdrant Cloud.

Obsługuje tworzenie kolekcji z dual vectors (dense + sparse),
upsert chunków i zarządzanie kolekcjami.
"""

import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from rich.console import Console

from config.settings import QdrantConfig, get_jina_config, get_qdrant_config
from src.chunking.base import Chunk

console = Console()

# Namespace dla deterministic UUID5
_UUID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class QdrantStore:
    """Klient Qdrant Cloud z obsługą dual vectors (dense + sparse)."""

    def __init__(self, config: QdrantConfig | None = None):
        cfg = config or get_qdrant_config()
        self.client = QdrantClient(
            url=cfg.url,
            api_key=cfg.api_key,
        )
        self._config = cfg
        self._dense_dim = get_jina_config().dimensions

    def collection_exists(self, name: str) -> bool:
        """Sprawdź czy kolekcja istnieje."""
        try:
            self.client.get_collection(name)
            return True
        except Exception:
            return False

    def create_collection(self, name: str, recreate: bool = False) -> None:
        """Utwórz kolekcję z konfiguracją dense + sparse vectors.

        Args:
            name: Nazwa kolekcji.
            recreate: Jeśli True, usuwa istniejącą kolekcję i tworzy nową.
        """
        if recreate and self.collection_exists(name):
            console.print(f"  Usuwanie istniejącej kolekcji: {name}")
            self.client.delete_collection(name)

        if self.collection_exists(name):
            console.print(f"  Kolekcja '{name}' już istnieje, pomijam tworzenie.")
            return

        self.client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._dense_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        console.print(f"  ✓ Utworzono kolekcję: {name}")

    def delete_collection(self, name: str) -> None:
        """Usuń kolekcję."""
        if self.collection_exists(name):
            self.client.delete_collection(name)
            console.print(f"  ✗ Usunięto kolekcję: {name}")

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[Any],
        batch_size: int = 64,
    ) -> None:
        """Wstaw chunki z dual vectors do kolekcji.

        Używa deterministic UUID5 — re-indeksowanie nadpisuje istniejące punkty.

        Args:
            collection_name: Nazwa kolekcji docelowej.
            chunks: Lista obiektów Chunk.
            dense_vectors: Lista wektorów dense (Jina).
            sparse_vectors: Lista SparseVector (BM25).
            batch_size: Rozmiar batcha do upsert.
        """
        if len(chunks) != len(dense_vectors) or len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"Niezgodna liczba: chunks={len(chunks)}, "
                f"dense={len(dense_vectors)}, sparse={len(sparse_vectors)}"
            )

        points = []
        for chunk, dense_vec, sparse_vec in zip(
            chunks, dense_vectors, sparse_vectors, strict=True
        ):
            # Deterministic ID — ten sam chunk_id = ten sam UUID
            point_id = str(uuid.uuid5(_UUID_NAMESPACE, chunk.chunk_id))

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec,
                    },
                    payload={
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        **chunk.metadata,
                    },
                )
            )

        # Upsert w batchach
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        console.print(
            f"  ✓ Upsert {len(points)} punktów do '{collection_name}'"
        )

    def create_dense_only_collection(self, name: str, recreate: bool = False) -> None:
        """Utwórz kolekcję z samym wektorem dense (bez sparse).

        Używane do embeddingów tytułów — nie potrzebują BM25.
        """
        if recreate and self.collection_exists(name):
            self.client.delete_collection(name)

        if self.collection_exists(name):
            return

        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self._dense_dim,
                distance=models.Distance.COSINE,
            ),
        )
        console.print(f"  ✓ Utworzono kolekcję (dense only): {name}")

    def upsert_titles(
        self,
        collection_name: str,
        titles: list[str],
        dense_vectors: list[list[float]],
        metadata_list: list[dict],
    ) -> None:
        """Wstaw tytuły z wektorami dense do kolekcji.

        Args:
            collection_name: Nazwa kolekcji.
            titles: Lista tytułów artykułów.
            dense_vectors: Lista wektorów dense (Jina 1024d).
            metadata_list: Lista metadanych per tytuł (source_url, slug itp.).
        """
        points = []
        for title, vec, meta in zip(titles, dense_vectors, metadata_list, strict=True):
            point_id = str(uuid.uuid5(_UUID_NAMESPACE, title))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={"title": title, **meta},
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)
        console.print(f"  ✓ Upsert {len(points)} tytułów do '{collection_name}'")

    def count_points(self, collection_name: str) -> int:
        """Zwróć liczbę punktów w kolekcji."""
        info = self.client.get_collection(collection_name)
        return info.points_count

    def close(self):
        """Zamknij klienta."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
