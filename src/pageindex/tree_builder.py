"""PageIndex — budowanie hierarchicznego drzewa semantycznego z artykułu.

Zamiast chunkingu + embeddingów, buduje drzewo sekcji (H1 → H2 → H3)
z generowanymi przez Claude streszczeniami. Query-time: reasoning po drzewie.
"""

import json
import re
import uuid
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config

SUMMARIZE_PROMPT = """Streść poniższą sekcję artykułu w 1-2 zdaniach (max 40 słów).
Skup się na KONKRECIE — co zawiera ta sekcja, jakie informacje można z niej wyciągnąć.

Nagłówek: {header}

Treść:
{content}

Streszczenie (tylko tekst, bez "Ta sekcja..."):"""


@dataclass
class TreeNode:
    """Węzeł w hierarchicznym drzewie PageIndex."""

    node_id: str
    title: str
    level: int  # 0=root, 1=H1, 2=H2, 3=H3
    summary: str = ""
    content: str = ""  # pełny tekst tylko dla leaf nodes
    children: list["TreeNode"] = field(default_factory=list)

    def to_dict(self, include_content: bool = False) -> dict:
        """Serializacja drzewa do dict (do wysłania do Claude)."""
        result = {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
        }
        if include_content and self.content:
            result["content"] = self.content
        if self.children:
            result["children"] = [c.to_dict(include_content) for c in self.children]
        return result


class TreeBuilder:
    """Buduje drzewo PageIndex z artykułu markdown."""

    def __init__(self):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    def _summarize(self, header: str, content: str) -> str:
        """Wygeneruj streszczenie sekcji przez Claude."""
        if len(content) < 100:
            return content[:200]  # krótkie sekcje — zwróć bez wywołania Claude

        response = self._client.messages.create(
            model=self._model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": SUMMARIZE_PROMPT.format(
                        header=header,
                        content=content[:3000],  # ogranicz kontekst
                    ),
                },
            ],
        )
        return response.content[0].text.strip()

    def build_from_markdown(self, markdown: str, title: str = "Artykuł") -> TreeNode:
        """Zbuduj drzewo z treści markdown.

        Args:
            markdown: Treść artykułu w markdown.
            title: Tytuł artykułu (root node).

        Returns:
            Root TreeNode z pełnym drzewem.
        """
        # Parsuj sekcje H2/H3
        sections = self._parse_sections(markdown)

        # Root
        root = TreeNode(
            node_id="root",
            title=title,
            level=0,
            summary=f"Artykuł: {title}",
        )

        # Buduj drzewo H2 → H3
        current_h2: TreeNode | None = None
        for sec in sections:
            if sec["level"] == 2:
                node = TreeNode(
                    node_id=str(uuid.uuid4())[:8],
                    title=sec["header"],
                    level=2,
                    content=sec["content"],
                )
                node.summary = self._summarize(sec["header"], sec["content"])
                root.children.append(node)
                current_h2 = node
            elif sec["level"] == 3 and current_h2:
                node = TreeNode(
                    node_id=str(uuid.uuid4())[:8],
                    title=sec["header"],
                    level=3,
                    content=sec["content"],
                )
                node.summary = self._summarize(sec["header"], sec["content"])
                current_h2.children.append(node)

        return root

    def _parse_sections(self, markdown: str) -> list[dict]:
        """Parsuj markdown na sekcje H2/H3 z treścią."""
        lines = markdown.split("\n")
        sections = []
        current_section = None

        for line in lines:
            h2_match = re.match(r"^##\s+(.+)$", line)
            h3_match = re.match(r"^###\s+(.+)$", line)

            if h2_match:
                if current_section:
                    current_section["content"] = "\n".join(current_section["content"]).strip()
                    sections.append(current_section)
                current_section = {
                    "level": 2,
                    "header": h2_match.group(1).strip(),
                    "content": [],
                }
            elif h3_match:
                if current_section:
                    current_section["content"] = "\n".join(current_section["content"]).strip()
                    sections.append(current_section)
                current_section = {
                    "level": 3,
                    "header": h3_match.group(1).strip(),
                    "content": [],
                }
            elif current_section:
                current_section["content"].append(line)

        if current_section:
            current_section["content"] = "\n".join(current_section["content"]).strip()
            sections.append(current_section)

        return sections

    def close(self):
        self._client.close()
