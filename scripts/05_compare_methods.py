"""Skrypt do porównania metod chunkingu na podstawie wyników ewaluacji.

Generuje tabelę porównawczą w terminalu i zapisuje CSV.

Użycie:
    python scripts/05_compare_methods.py
    python scripts/05_compare_methods.py --results-file data/results/evaluation_20260403_120000.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table

from config.settings import get_paths

console = Console()


def _znajdz_najnowszy_wynik(results_dir: Path) -> Path | None:
    """Znajdź najnowszy plik evaluation_*.json."""
    pliki = sorted(results_dir.glob("evaluation_*.json"), reverse=True)
    return pliki[0] if pliki else None


def porownaj_metody(results_file: Path | None = None) -> None:
    """Wyświetl tabelę porównawczą metod chunkingu.

    Args:
        results_file: Ścieżka do pliku JSON z wynikami.
                      Domyślnie najnowszy plik w data/results/.
    """
    # Znajdź plik wyników
    if results_file is None:
        results_dir = get_paths().results
        results_file = _znajdz_najnowszy_wynik(results_dir)
        if results_file is None:
            console.print(
                "[red]Brak plików z wynikami w data/results/. "
                "Uruchom najpierw 04_run_evaluation.py[/red]"
            )
            return

    console.print(f"[dim]Plik wyników: {results_file}[/dim]\n")

    data = json.loads(results_file.read_text(encoding="utf-8"))

    # --- Tabela porównawcza ---
    table = Table(title="Porównanie metod chunkingu (RAGAS)")
    table.add_column("Metoda", style="cyan bold")

    # Zbierz wszystkie metryki
    all_metrics = set()
    for method_data in data.values():
        scores = method_data.get("scores", {})
        all_metrics.update(scores.keys())

    all_metrics = sorted(all_metrics)
    for metric in all_metrics:
        table.add_column(metric, justify="right")

    # Dodaj wiersze z podświetleniem najlepszych
    best_per_metric = {}
    for metric in all_metrics:
        values = {
            method: d.get("scores", {}).get(metric, 0)
            for method, d in data.items()
        }
        if values:
            best_per_metric[metric] = max(values, key=values.get)

    for method, method_data in data.items():
        scores = method_data.get("scores", {})
        if method_data.get("error"):
            row = [method] + [f"[red]BŁĄD[/red]"] * len(all_metrics)
        else:
            row = [method]
            for metric in all_metrics:
                val = scores.get(metric, 0)
                formatted = f"{val:.4f}"
                if best_per_metric.get(metric) == method:
                    formatted = f"[bold green]{formatted} ★[/bold green]"
                row.append(formatted)
        table.add_row(*row)

    console.print(table)
    console.print("[dim]★ = najlepszy wynik w danej metryce[/dim]\n")

    # --- Zapisz CSV ---
    csv_data = []
    for method, method_data in data.items():
        row = {"metoda": method}
        row.update(method_data.get("scores", {}))
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    csv_path = results_file.parent / results_file.stem.replace("evaluation", "comparison") + ".csv"

    # Poprawka: pathlib nie obsługuje + na Path
    csv_path = results_file.parent / f"{results_file.stem.replace('evaluation', 'comparison')}.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]✓ Zapisano CSV: {csv_path}[/green]")

    # --- Podsumowanie per pytanie (jeśli dostępne) ---
    for method, method_data in data.items():
        per_q = method_data.get("per_question", [])
        if per_q:
            console.print(f"\n[bold]Szczegóły: {method}[/bold]")
            q_table = Table(show_lines=True)
            q_table.add_column("Pytanie", max_width=50)
            for metric in all_metrics:
                q_table.add_column(metric, justify="right", max_width=12)

            for entry in per_q:
                row = [str(entry.get("user_input", ""))[:50]]
                for metric in all_metrics:
                    val = entry.get(metric, "")
                    if isinstance(val, (int, float)):
                        row.append(f"{val:.3f}")
                    else:
                        row.append(str(val))
                q_table.add_row(*row)

            console.print(q_table)


def main():
    parser = argparse.ArgumentParser(description="Porównanie metod chunkingu")
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Ścieżka do pliku JSON z wynikami ewaluacji",
    )
    args = parser.parse_args()

    results_file = Path(args.results_file) if args.results_file else None
    porownaj_metody(results_file)


if __name__ == "__main__":
    main()
