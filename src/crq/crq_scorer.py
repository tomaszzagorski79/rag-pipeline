"""CRQ Scoring — Content Retrieval Quality.

Ocena jakości artykułu pod kątem retrieval: Information Density,
BLUF compliance, Chunking Quality, EAV coverage.
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config

DENSITY_PROMPT = """Oceń gęstość informacyjną poniższego tekstu artykułu.
Gęstość informacyjna to stosunek faktów/konkretów do "puchu" (ogólników, słów modalnych, wstępów).

Zwróć JSON:
{{
  "score": 0-100,
  "facts_count": liczba konkretnych faktów/danych,
  "filler_examples": ["przykład ogólnika 1", "przykład ogólnika 2"],
  "good_examples": ["przykład konkretu 1", "przykład konkretu 2"],
  "recommendation": "jedna rekomendacja jak poprawić"
}}

Tekst:
{text}
"""

BLUF_PROMPT = """Oceń czy każda sekcja H2 zaczyna się od kluczowej informacji (BLUF — Bottom Line Up Front).
BLUF = w pierwszych 50 słowach sekcji jest konkretna odpowiedź/wniosek, nie ogólne wprowadzenie.

Sekcje artykułu:
{sections}

Zwróć JSON array:
[
  {{"header": "tytuł H2", "has_bluf": true/false, "first_50_words": "...", "recommendation": "..."}}
]
"""

EAV_PROMPT = """Wyciągnij trójki EAV (Entity-Attribute-Value) z poniższego tekstu.
EAV to fakty w formacie: Encja — Atrybut — Wartość.
Przykład: "VAT w Niemczech — stawka podstawowa — 19%"

Zwróć JSON:
{{
  "triples": [
    {{"entity": "...", "attribute": "...", "value": "..."}},
  ],
  "density_per_1000_words": liczba,
  "score": 0-100,
  "recommendation": "..."
}}

Tekst:
{text}
"""


@dataclass
class CRQArticleScore:
    """Wynik CRQ dla jednego artykułu."""

    article_slug: str
    article_title: str
    information_density: int = 0
    bluf_compliance: int = 0
    chunking_quality: int = 0
    eav_coverage: int = 0
    overall_crq: int = 0
    density_details: dict = field(default_factory=dict)
    bluf_details: list = field(default_factory=list)
    chunking_details: dict = field(default_factory=dict)
    eav_details: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)


class CRQScorer:
    """Scorer CRQ — ocenia jakość artykułu pod kątem retrieval."""

    def __init__(self):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    def _call_claude(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse_json(self, raw: str) -> dict | list:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)

    def _score_density(self, text: str) -> dict:
        """Oceń gęstość informacyjną (Claude)."""
        raw = self._call_claude(DENSITY_PROMPT.format(text=text[:8000]))
        try:
            return self._parse_json(raw)
        except json.JSONDecodeError:
            return {"score": 50, "recommendation": "Nie udało się przeanalizować."}

    def _score_bluf(self, markdown_text: str) -> tuple[int, list]:
        """Oceń BLUF compliance per sekcja H2."""
        # Wyciągnij sekcje H2
        sections = []
        current_header = ""
        current_text = []

        for line in markdown_text.split("\n"):
            if line.startswith("## "):
                if current_header:
                    sections.append({
                        "header": current_header,
                        "text": " ".join(current_text)[:200],
                    })
                current_header = line.lstrip("# ").strip()
                current_text = []
            elif current_header:
                if line.strip():
                    current_text.append(line.strip())

        if current_header:
            sections.append({
                "header": current_header,
                "text": " ".join(current_text)[:200],
            })

        if not sections:
            return 0, []

        formatted = "\n".join(
            f"## {s['header']}\n{s['text']}" for s in sections
        )

        raw = self._call_claude(BLUF_PROMPT.format(sections=formatted))
        try:
            details = self._parse_json(raw)
            has_bluf = sum(1 for d in details if d.get("has_bluf"))
            score = int(has_bluf / len(details) * 100) if details else 0
            return score, details
        except (json.JSONDecodeError, TypeError):
            return 50, []

    def _score_chunking_quality(self, markdown_text: str) -> dict:
        """Oceń jakość chunkingu (programatycznie, bez AI)."""
        from src.chunking.header_based import HeaderChunker

        chunker = HeaderChunker()
        chunks = chunker.chunk(markdown_text, {"slug": "test"})

        if not chunks:
            return {"score": 0, "num_chunks": 0}

        lengths = [len(c.text) for c in chunks]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        orphans = sum(1 for l in lengths if l < 100)

        # Score: penalizuj za dużą wariancję i orphans
        score = 100
        if variance > 500000:  # wysoka wariancja
            score -= 20
        if orphans > 0:
            score -= orphans * 10
        if avg_len > 2000:
            score -= 15
        if avg_len < 200:
            score -= 15

        score = max(0, min(100, score))

        return {
            "score": score,
            "num_chunks": len(chunks),
            "avg_length": int(avg_len),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "orphan_chunks": orphans,
            "variance": int(variance),
        }

    def _score_eav(self, text: str) -> dict:
        """Oceń pokrycie EAV (Claude)."""
        raw = self._call_claude(EAV_PROMPT.format(text=text[:8000]))
        try:
            return self._parse_json(raw)
        except json.JSONDecodeError:
            return {"score": 50, "triples": [], "recommendation": "Nie udało się przeanalizować."}

    def score_article(
        self, markdown_text: str, slug: str, title: str
    ) -> CRQArticleScore:
        """Pełna ocena CRQ artykułu.

        Args:
            markdown_text: Treść artykułu (bez frontmatter).
            slug: Identyfikator artykułu.
            title: Tytuł artykułu.

        Returns:
            CRQArticleScore z 4 wymiarami i rekomendacjami.
        """
        result = CRQArticleScore(article_slug=slug, article_title=title)

        # 1. Information Density (Claude)
        density = self._score_density(markdown_text)
        result.information_density = density.get("score", 50)
        result.density_details = density

        # 2. BLUF compliance (Claude)
        bluf_score, bluf_details = self._score_bluf(markdown_text)
        result.bluf_compliance = bluf_score
        result.bluf_details = bluf_details

        # 3. Chunking Quality (programatycznie)
        chunking = self._score_chunking_quality(markdown_text)
        result.chunking_quality = chunking.get("score", 50)
        result.chunking_details = chunking

        # 4. EAV coverage (Claude)
        eav = self._score_eav(markdown_text)
        result.eav_coverage = eav.get("score", 50)
        result.eav_details = eav

        # Overall CRQ (średnia ważona)
        result.overall_crq = int(
            result.information_density * 0.3
            + result.bluf_compliance * 0.25
            + result.chunking_quality * 0.2
            + result.eav_coverage * 0.25
        )

        # Zbierz rekomendacje
        for key, details in [
            ("density", density),
            ("eav", eav),
        ]:
            rec = details.get("recommendation")
            if rec:
                result.recommendations.append(f"[{key}] {rec}")

        for bluf in bluf_details:
            if not bluf.get("has_bluf") and bluf.get("recommendation"):
                result.recommendations.append(
                    f"[BLUF: {bluf.get('header', '?')}] {bluf['recommendation']}"
                )

        return result

    def close(self):
        self._client.close()
