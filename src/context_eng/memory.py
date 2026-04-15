"""Memory wielowarstwowa — short-term (sesja) + episodic (persystentna).

Komponent 5 Context Engineering:
- Short-term: historia bieżącej sesji (in-memory)
- Episodic: zapisane interakcje z poprzednich sesji (JSON na dysku)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class MemoryEntry:
    """Pojedyncza interakcja w pamięci."""

    query: str
    answer: str
    timestamp: str
    intent: str = ""
    sources_used: list[str] = field(default_factory=list)
    num_contexts: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "sources_used": self.sources_used,
            "num_contexts": self.num_contexts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            timestamp=data.get("timestamp", ""),
            intent=data.get("intent", ""),
            sources_used=data.get("sources_used", []),
            num_contexts=data.get("num_contexts", 0),
        )


class MultiLayerMemory:
    """Pamięć wielowarstwowa — short-term (sesja) + episodic (JSON)."""

    def __init__(self, memory_dir: Path | None = None, session_id: str | None = None):
        """Inicjalizacja.

        Args:
            memory_dir: Katalog dla episodic memory (domyślnie data/memory/).
            session_id: ID sesji (domyślnie timestamp).
        """
        from config.settings import get_paths

        self._memory_dir = memory_dir or (get_paths().data / "memory")
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        self._session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._short_term: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        """Dodaj wpis do short-term i zapisz do episodic."""
        self._short_term.append(entry)
        self._persist_episodic(entry)

    def get_short_term(self, limit: int = 10) -> list[MemoryEntry]:
        """Zwróć ostatnie N wpisów z bieżącej sesji."""
        return self._short_term[-limit:]

    def get_session_context(self, max_entries: int = 3) -> str:
        """Zwróć sformatowany kontekst sesji dla LLM.

        Args:
            max_entries: Ile ostatnich interakcji dołączyć.

        Returns:
            Tekst z historią sesji (do dodania do kontekstu).
        """
        recent = self.get_short_term(limit=max_entries)
        if not recent:
            return ""

        parts = ["--- Historia bieżącej sesji ---"]
        for i, entry in enumerate(recent, 1):
            parts.append(f"[{i}] Pytanie: {entry.query}")
            parts.append(f"    Odpowiedź: {entry.answer[:200]}...")
        return "\n".join(parts)

    def search_episodic(self, query_substring: str, limit: int = 5) -> list[MemoryEntry]:
        """Przeszukaj episodic memory po podłańcuchu w query.

        Args:
            query_substring: Fragment tekstu do wyszukania.
            limit: Max wyników.

        Returns:
            Lista MemoryEntry z poprzednich sesji.
        """
        results = []
        query_lower = query_substring.lower()

        for session_file in sorted(self._memory_dir.glob("session_*.json"), reverse=True):
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                for entry_dict in data.get("entries", []):
                    if query_lower in entry_dict.get("query", "").lower():
                        results.append(MemoryEntry.from_dict(entry_dict))
                        if len(results) >= limit:
                            return results
            except (json.JSONDecodeError, FileNotFoundError):
                continue

        return results

    def _persist_episodic(self, entry: MemoryEntry) -> None:
        """Zapisz wpis do pliku JSON sesji."""
        session_file = self._memory_dir / f"session_{self._session_id}.json"

        if session_file.exists():
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {"session_id": self._session_id, "entries": []}
        else:
            data = {"session_id": self._session_id, "entries": []}

        data["entries"].append(entry.to_dict())

        session_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_short_term(self) -> None:
        """Wyczyść short-term memory (ale zachowaj episodic)."""
        self._short_term = []

    def get_stats(self) -> dict:
        """Statystyki pamięci."""
        sessions = list(self._memory_dir.glob("session_*.json"))
        total_episodic = 0
        for s in sessions:
            try:
                data = json.loads(s.read_text(encoding="utf-8"))
                total_episodic += len(data.get("entries", []))
            except json.JSONDecodeError:
                pass
        return {
            "short_term_entries": len(self._short_term),
            "episodic_sessions": len(sessions),
            "episodic_total_entries": total_episodic,
        }
