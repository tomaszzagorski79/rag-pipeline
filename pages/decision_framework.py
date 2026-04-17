"""Strona Decision Framework — jaki RAG kiedy? Drzewo decyzyjne."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("22. Decision Framework")
    st.markdown("**Jaki RAG, kiedy?** Praktyczny model decyzyjny na bazie benchmarków 2024-2026.")

    # Zasada #1
    st.header("Zasada #1: Zawsze zaczynaj od Hybrid + Rerank")
    st.info("""
**T2-RAGBench (2026):** Hybrid (BM25 + dense) + Cohere Rerank = **Recall@5 = 0.816**
— to +39% vs vanilla dense, +17% vs Hybrid RRF bez rerankera.

Wszystkie inne techniki powinny być oceniane **względem tego baseline'u**.
    """)

    # Drzewo decyzyjne
    st.header("Drzewo decyzyjne")

    decision_data = [
        {
            "use_case": "Proste FAQ, krótka dokumentacja (<10k chunków)",
            "rekomendacja": "Naive lub Advanced RAG + rerank",
            "alternatywa": "Hybrid jeśli dużo nazw własnych",
            "zakladka": "4-5",
        },
        {
            "use_case": "Korporacyjna baza wiedzy, dokumentacja techniczna",
            "rekomendacja": "Advanced RAG: Hybrid + semantic chunking + rerank",
            "alternatywa": "+ CRAG jeśli jakość źródeł niestabilna",
            "zakladka": "3-5-8",
        },
        {
            "use_case": "Query–document gap (eksploracja, research)",
            "rekomendacja": "HyDE + Hybrid",
            "alternatywa": "RAG-Fusion dla wielu perspektyw",
            "zakladka": "6-18",
        },
        {
            "use_case": "Dokumenty fact-heavy (finanse, medycyna, prawo)",
            "rekomendacja": "Hybrid + Rerank (NIE HyDE — halucynuje liczby)",
            "alternatywa": "CRAG z fallbackiem",
            "zakladka": "5-8",
        },
        {
            "use_case": "Multi-hop QA, porównania, pytania globalne",
            "rekomendacja": "GraphRAG lub RAPTOR",
            "alternatywa": "Agentic RAG z wieloma narzędziami",
            "zakladka": "15-20-14",
        },
        {
            "use_case": "Długie dokumenty z hierarchią (podręczniki, raporty)",
            "rekomendacja": "RAPTOR (collapsed tree) lub PageIndex",
            "alternatywa": "GraphRAG jeśli ważne relacje encji",
            "zakladka": "20-13",
        },
        {
            "use_case": "Chatbot real-time z niską latencją",
            "rekomendacja": "Naive/Advanced + cache",
            "alternatywa": "Speculative RAG",
            "zakladka": "4-21",
        },
        {
            "use_case": "Złożone workflowy, multi-step reasoning",
            "rekomendacja": "Agentic RAG z ReAct",
            "alternatywa": "Adaptive RAG (routing)",
            "zakladka": "14-7",
        },
        {
            "use_case": "Max faithfulness, zero halucynacji",
            "rekomendacja": "CRAG + Hallucination Detection + FLARE",
            "alternatywa": "Speculative RAG z weryfikatorem",
            "zakladka": "8-10-19",
        },
        {
            "use_case": "Zróżnicowane zapytania, różna trudność",
            "rekomendacja": "Adaptive RAG (routing)",
            "alternatywa": "Context Engineering (pełny framework)",
            "zakladka": "7-17",
        },
    ]

    import pandas as pd

    df = pd.DataFrame(decision_data)
    df.columns = ["Use case", "Rekomendacja", "Alternatywa", "Zakładki"]
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Zasada #2
    st.header("Zasada #2: Nie dobieraj architektury przed ewaluacją")
    st.markdown("""
1. Zbuduj **Naive RAG** w 1 dzień ✅ (masz — zakładka 4)
2. Zbuduj **test set** 50-100 pytań ✅ (masz — zakładka 9)
3. Odpal **RAGAS** — zobacz który komponent zawodzi ✅ (masz)
4. Dodaj najtańszą poprawkę (**rerank → hybrid → chunking**) ✅ (masz)
5. Re-ewaluuj. Gdy plateau — rozważ strukturalną zmianę (GraphRAG, Agentic)
    """)

    # Zasada #3: Koszty
    st.header("Zasada #3: Koszty (orientacyjne per 1000 zapytań)")

    cost_data = [
        ("Naive RAG", "$1–3", "1 embedding + 1 LLM", "4"),
        ("Advanced + Rerank", "$3–8", "+ re-ranker model", "5"),
        ("HyDE / RAG-Fusion", "$5–15", "2-5 dodatkowych LLM calls", "6 / 18"),
        ("CRAG", "$8–15", "+ evaluator + fallback", "8"),
        ("FLARE", "$10–25", "N × retrieval w trakcie generacji", "19"),
        ("Agentic RAG (5-10 kroków)", "$30–150", "5-20× droższe niż Naive", "14"),
        ("GraphRAG indeksacja", "$100–10000", "jednorazowo, LLM per chunk", "15"),
    ]

    df_cost = pd.DataFrame(cost_data, columns=["Architektura", "Koszt/1k zapytań", "Składowe", "Zakładka"])
    st.dataframe(df_cost, use_container_width=True, hide_index=True)

    # Mapowanie metryk na komponenty
    st.header("Która metryka RAGAS → jaka poprawka?")

    fix_data = [
        ("Retriever", "Context Precision, Context Recall", "Zmień embeddingi, chunking, dodaj rerankera, Hybrid/HyDE"),
        ("Generator", "Faithfulness, Answer Relevancy", "Zmień LLM, prompt engineering, Self-RAG/CRAG"),
        ("End-to-end", "Answer Correctness, Semantic Similarity", "Zmiany architektoniczne, routing, Adaptive/Agentic"),
    ]

    df_fix = pd.DataFrame(fix_data, columns=["Komponent", "Metryki RAGAS", "Poprawka"])
    st.dataframe(df_fix, use_container_width=True, hide_index=True)

    # Pipeline w Twoim narzędziu
    st.header("Twój pipeline — dostępne techniki")

    techniques = [
        ("4", "Naive RAG", "Hybrid search (BM25+vector+RRF)", "✅"),
        ("5", "Re-ranking", "FlashRank cross-encoder", "✅"),
        ("6", "HyDE", "Hipotetyczna odpowiedź → search", "✅"),
        ("7", "Adaptive RAG", "Routing: no_retrieval/simple/medium/complex", "✅"),
        ("8", "CRAG", "Korekcja + knowledge strip + reformulate", "✅"),
        ("10", "Hallucination Detection", "Claim-by-claim verification", "✅"),
        ("13", "PageIndex", "Vectorless reasoning-based", "✅"),
        ("14", "Agentic RAG", "ReAct + tool_use (4 narzędzia)", "✅"),
        ("15", "Graph RAG", "Neo4j + EAV triples", "✅"),
        ("17", "Context Engineering", "6 komponentów: orchestrator + memory + facets", "✅"),
        ("18", "RAG-Fusion", "Multi-query + RRF", "✅"),
        ("19", "FLARE", "Active retrieval w trakcie generacji", "✅"),
        ("20", "RAPTOR", "Hierarchiczny indeks drzewiasty", "✅"),
        ("21", "Speculative RAG", "Drafter (Haiku) + Verifier (Sonnet)", "✅"),
    ]

    df_tech = pd.DataFrame(techniques, columns=["Zakładka", "Technika", "Opis", "Status"])
    st.dataframe(df_tech, use_container_width=True, hide_index=True)

    # PEŁNA TABELA PORÓWNAWCZA
    st.markdown("---")
    st.header("Pełna tabela porównawcza — kiedy który RAG?")

    comparison = [
        {
            "RAG": "Naive (Hybrid Search)",
            "Zakładka": "4",
            "Stosować gdy": "FAQ, prosta dokumentacja, szybki baseline",
            "NIE stosować gdy": "Złożone pytania, multi-hop, porównania",
            "Latencja": "⚡ Niska",
            "Koszt": "💰",
            "Recall": "⭐⭐",
        },
        {
            "RAG": "Re-ranking",
            "Zakładka": "5",
            "Stosować gdy": "Zawsze jako pierwszy upgrade (+39% Recall@5)",
            "NIE stosować gdy": "Bardzo mała baza (<50 chunków)",
            "Latencja": "⚡ Niska (+200ms)",
            "Koszt": "💰",
            "Recall": "⭐⭐⭐⭐",
        },
        {
            "RAG": "HyDE",
            "Zakładka": "6",
            "Stosować gdy": "Eksploracja, research, query-document gap",
            "NIE stosować gdy": "Fakty/liczby (halucynuje), niska latencja, finanse/prawo",
            "Latencja": "🐌 Średnia",
            "Koszt": "💰💰",
            "Recall": "⭐⭐⭐",
        },
        {
            "RAG": "Adaptive RAG",
            "Zakładka": "7",
            "Stosować gdy": "Zróżnicowane pytania, optymalizacja kosztów",
            "NIE stosować gdy": "Jednolita trudność pytań, potrzeba determinizmu",
            "Latencja": "⚡-🐌 Zależy",
            "Koszt": "💰-💰💰💰",
            "Recall": "⭐⭐⭐",
        },
        {
            "RAG": "CRAG",
            "Zakładka": "8",
            "Stosować gdy": "Niestabilne źródła, krytyczne zastosowania (fail gracefully)",
            "NIE stosować gdy": "Dobry indeks (hybrid+rerank lepszy), niska latencja",
            "Latencja": "🐌 Średnia",
            "Koszt": "💰💰",
            "Recall": "⭐⭐⭐",
        },
        {
            "RAG": "RAG-Fusion",
            "Zakładka": "18",
            "Stosować gdy": "Pytania szerokie, research, wiele perspektyw",
            "NIE stosować gdy": "Pytania wąskie/faktologiczne, ograniczony budżet",
            "Latencja": "🐌 Średnia (1.7x)",
            "Koszt": "💰💰",
            "Recall": "⭐⭐⭐⭐",
        },
        {
            "RAG": "FLARE",
            "Zakładka": "19",
            "Stosować gdy": "Długie odpowiedzi, multi-hop, max faithfulness",
            "NIE stosować gdy": "Krótkie odpowiedzi, niska latencja, FAQ",
            "Latencja": "🐌🐌 Wysoka",
            "Koszt": "💰💰💰",
            "Recall": "⭐⭐⭐⭐",
        },
        {
            "RAG": "PageIndex",
            "Zakładka": "13",
            "Stosować gdy": "Długie strukturalne dokumenty, wyjaśnialność, audit trail",
            "NIE stosować gdy": "Duży korpus (>100 docs), krótkie dokumenty, niska latencja",
            "Latencja": "🐌 Średnia",
            "Koszt": "💰💰",
            "Recall": "⭐⭐⭐⭐",
        },
        {
            "RAG": "RAPTOR",
            "Zakładka": "20",
            "Stosować gdy": "Hierarchiczne dokumenty, multi-level QA, podsumowania",
            "NIE stosować gdy": "Mały korpus, brak hierarchii, kosztowna indeksacja",
            "Latencja": "⚡ Niska (po budowie)",
            "Koszt": "💰💰💰 (indeksacja)",
            "Recall": "⭐⭐⭐⭐⭐",
        },
        {
            "RAG": "Graph RAG",
            "Zakładka": "15",
            "Stosować gdy": "Multi-hop, relacje encji, pytania globalne, topical authority",
            "NIE stosować gdy": "Mały korpus, brak encji/relacji, kosztowna indeksacja",
            "Latencja": "🐌 Średnia",
            "Koszt": "💰💰💰 (indeksacja)",
            "Recall": "⭐⭐⭐⭐",
        },
        {
            "RAG": "Speculative RAG",
            "Zakładka": "21",
            "Stosować gdy": "Porównanie perspektyw, szybszy niż standard (drafter tani)",
            "NIE stosować gdy": "Proste pytania, max dokładność, ograniczony budżet",
            "Latencja": "⚡ Niska",
            "Koszt": "💰💰",
            "Recall": "⭐⭐⭐",
        },
        {
            "RAG": "Agentic RAG",
            "Zakładka": "14",
            "Stosować gdy": "Złożone workflowy, multi-step, research, due diligence",
            "NIE stosować gdy": "Proste pytania (5-20x droższe!), niska latencja",
            "Latencja": "🐌🐌 Wysoka",
            "Koszt": "💰💰💰💰",
            "Recall": "⭐⭐⭐⭐⭐",
        },
        {
            "RAG": "Context Engineering",
            "Zakładka": "17",
            "Stosować gdy": "Produkcyjne systemy, wiele źródeł, budżetowanie tokenów",
            "NIE stosować gdy": "Prototyp, jedno źródło, brak złożoności",
            "Latencja": "🐌🐌 Wysoka",
            "Koszt": "💰💰💰💰",
            "Recall": "⭐⭐⭐⭐⭐",
        },
    ]

    df_compare = pd.DataFrame(comparison)
    st.dataframe(df_compare, use_container_width=True, hide_index=True, height=550)

    # Scatter koszty × jakość
    st.markdown("---")
    st.header("Scatter: Koszty × Jakość")
    st.caption("Oś X: koszt (1-4 💰), Oś Y: jakość retrieval (1-5 ⭐), wielkość: latencja")

    # Parsowanie z tabeli
    cost_map = {"💰": 1, "💰💰": 2, "💰💰💰": 3, "💰💰💰💰": 4}
    lat_map = {
        "⚡ Niska": 1, "⚡ Niska (+200ms)": 1, "⚡-🐌 Zależy": 2,
        "⚡ Niska (po budowie)": 1,
        "🐌 Średnia": 2, "🐌 Średnia (1.7x)": 2, "🐌🐌 Wysoka": 4,
    }

    sc_rows = []
    for c in comparison:
        cost_base = c["Koszt"].split(" ")[0].strip()
        cost = cost_map.get(cost_base, 2)
        quality = c["Recall"].count("⭐")
        lat = lat_map.get(c["Latencja"], 2)
        sc_rows.append({
            "rag": c["RAG"],
            "cost": cost,
            "quality": quality,
            "latency": lat * 15 + 10,  # rozmiar
            "tab": c["Zakładka"],
        })

    df_sc = pd.DataFrame(sc_rows)
    import plotly.express as px

    fig_tradeoff = px.scatter(
        df_sc, x="cost", y="quality",
        size="latency", color="rag",
        text="rag", hover_data=["tab"],
        labels={"cost": "Koszt (💰)", "quality": "Jakość (⭐)"},
        title="Trade-off: koszt vs jakość (wielkość = latencja)",
    )
    fig_tradeoff.update_traces(textposition="top center")
    fig_tradeoff.update_layout(height=550, showlegend=False, xaxis_range=[0.5, 4.5], yaxis_range=[1, 6])
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    st.caption("💡 **Sweet spot:** prawy-dolny róg (wysoka jakość, niski koszt). Lewy-górny = przepłacasz.")
