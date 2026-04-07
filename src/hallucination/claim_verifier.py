"""Detekcja halucynacji — weryfikacja twierdzeń w odpowiedzi vs konteksty.

Claude wyciąga twierdzenia faktyczne z odpowiedzi, potem weryfikuje
każde osobno względem dostarczonych kontekstów.
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config

EXTRACT_CLAIMS_PROMPT = """Wyciągnij WSZYSTKIE twierdzenia faktyczne z poniższej odpowiedzi.
Zwróć JSON array z listą twierdzeń (krótkie, jedno zdanie każde).
Zwróć TYLKO JSON array, bez komentarzy.

Odpowiedź:
{answer}

Format: ["twierdzenie 1", "twierdzenie 2", ...]
"""

VERIFY_CLAIM_PROMPT = """Sprawdź czy poniższe twierdzenie jest poparte przez dostarczone konteksty.

TWIERDZENIE: {claim}

KONTEKSTY:
{contexts}

Odpowiedz w formacie JSON (bez komentarzy):
{{
  "supported": true/false,
  "confidence": 0.0-1.0,
  "evidence": "cytat lub wyjaśnienie dlaczego poparte/niepoparte"
}}
"""


@dataclass
class ClaimVerification:
    """Weryfikacja pojedynczego twierdzenia."""

    claim: str
    supported: bool
    confidence: float
    evidence: str


@dataclass
class HallucinationReport:
    """Raport z detekcji halucynacji."""

    answer: str
    claims: list[ClaimVerification] = field(default_factory=list)
    overall_score: float = 0.0  # % popartych twierdzeń
    num_supported: int = 0
    num_unsupported: int = 0


class ClaimVerifier:
    """Weryfikator twierdzeń — wykrywa halucynacje w odpowiedziach RAG."""

    def __init__(self):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    def _call_claude(self, prompt: str, max_tokens: int = 1000) -> str:
        """Wywołaj Claude z promptem."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def extract_claims(self, answer: str) -> list[str]:
        """Wyciągnij twierdzenia faktyczne z odpowiedzi.

        Args:
            answer: Tekst odpowiedzi RAG.

        Returns:
            Lista twierdzeń (stringi).
        """
        raw = self._call_claude(EXTRACT_CLAIMS_PROMPT.format(answer=answer))

        # Parsuj JSON — Claude czasem dodaje markdown ```json
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            claims = json.loads(raw)
            if isinstance(claims, list):
                return [str(c) for c in claims]
        except json.JSONDecodeError:
            pass

        # Fallback: split po nowych liniach
        return [l.strip().lstrip("- ") for l in raw.split("\n") if l.strip()]

    def verify_claim(self, claim: str, contexts: list[str]) -> ClaimVerification:
        """Zweryfikuj jedno twierdzenie względem kontekstów.

        Args:
            claim: Twierdzenie do weryfikacji.
            contexts: Lista fragmentów kontekstu.

        Returns:
            ClaimVerification z wynikiem.
        """
        formatted_ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        raw = self._call_claude(
            VERIFY_CLAIM_PROMPT.format(claim=claim, contexts=formatted_ctx),
            max_tokens=300,
        )

        # Parsuj JSON
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            data = json.loads(raw)
            return ClaimVerification(
                claim=claim,
                supported=bool(data.get("supported", False)),
                confidence=float(data.get("confidence", 0.5)),
                evidence=str(data.get("evidence", "")),
            )
        except (json.JSONDecodeError, KeyError):
            return ClaimVerification(
                claim=claim,
                supported=False,
                confidence=0.0,
                evidence=f"Błąd parsowania odpowiedzi: {raw[:100]}",
            )

    def verify_answer(
        self,
        answer: str,
        contexts: list[str],
    ) -> HallucinationReport:
        """Pełna weryfikacja odpowiedzi: ekstrakcja twierdzeń + weryfikacja każdego.

        Args:
            answer: Tekst odpowiedzi RAG.
            contexts: Lista fragmentów kontekstu użytych do generowania.

        Returns:
            HallucinationReport ze szczegółami per twierdzenie.
        """
        # 1. Wyciągnij twierdzenia
        claims = self.extract_claims(answer)

        # 2. Weryfikuj każde
        verifications = []
        for claim in claims:
            v = self.verify_claim(claim, contexts)
            verifications.append(v)

        # 3. Oblicz score
        num_supported = sum(1 for v in verifications if v.supported)
        num_unsupported = len(verifications) - num_supported
        overall = (num_supported / len(verifications) * 100) if verifications else 0

        return HallucinationReport(
            answer=answer,
            claims=verifications,
            overall_score=overall,
            num_supported=num_supported,
            num_unsupported=num_unsupported,
        )

    def close(self):
        self._client.close()
