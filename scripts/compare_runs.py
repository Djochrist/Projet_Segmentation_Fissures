"""
Compare les métriques de plusieurs évaluations (runs) pour voir facilement
si les changements apportés au modèle/entraînement l'améliorent réellement.

Fonctionne avec les rapports JSON produits par `scripts/evaluate.py`
(fichiers `{model_type}_eval_{split}.json`, contenant {"metrics": {...}}).

Usage typique :
    # Après chaque évaluation, évitez d'écraser le rapport précédent en
    # utilisant un --output-dir différent par run, par exemple :
    #   python scripts/evaluate.py --model runs/v1/model_final.pth \
    #       --data-root data --output-dir outputs/evaluation/v1_lr0001
    #   python scripts/evaluate.py --model runs/v2/model_final.pth \
    #       --data-root data --output-dir outputs/evaluation/v2_lr001
    #
    # Puis comparez tous les runs trouvés sous outputs/evaluation :
    python scripts/compare_runs.py --results-dir outputs/evaluation

    # Ou listez explicitement les rapports à comparer :
    python scripts/compare_runs.py \
        outputs/evaluation/v1_lr0001/maskrcnn_eval_test.json \
        outputs/evaluation/v2_lr001/maskrcnn_eval_test.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="*", help="Chemins explicites vers des rapports JSON à comparer")
    parser.add_argument("--results-dir", type=str, default=None,
                         help="Dossier à parcourir récursivement pour trouver tous les *_eval_*.json")
    parser.add_argument("--output-dir", type=str, default="outputs/comparisons",
                         help="Dossier où sauvegarder le tableau CSV et le graphique")
    parser.add_argument("--metric", type=str, default=None,
                         help="Nom d'une métrique précise à tracer en graphique (ex: box_map50). "
                              "Par défaut : trace les métriques principales (mAP box/mask).")
    parser.add_argument("--no-plot", action="store_true", help="Ne pas générer de graphique (seulement le tableau)")
    return parser.parse_args()


def find_reports(reports_args, results_dir):
    paths = [Path(p) for p in reports_args]
    if results_dir:
        rd = Path(results_dir)
        if not rd.exists():
            sys.exit(f"❌ Dossier introuvable : {rd}")
        paths.extend(sorted(rd.rglob("*_eval_*.json")))
    if not paths:
        sys.exit("❌ Aucun rapport trouvé. Précisez des fichiers ou --results-dir.")
    seen = set()
    unique_paths = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen and rp.exists():
            seen.add(rp)
            unique_paths.append(p)
    return unique_paths


def run_label(path: Path) -> str:
    """Nom du run = nom du dossier parent (là où vous mettez --output-dir lors
    de l'évaluation), pour identifier facilement quel run est quoi."""
    parent = path.parent.name
    if parent and parent not in ("evaluation", "."):
        return parent
    return path.stem


def load_reports(paths):
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠ Rapport ignoré (illisible) : {p} ({e})")
            continue
        metrics = report.get("metrics", {})
        row = {
            "run": run_label(p),
            "model_type": report.get("model_type", "?"),
            "split": report.get("split", "?"),
            "fichier": str(p),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def print_table(df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("  Comparaison des runs")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        cols = [c for c in df.columns if c != "fichier"]
        print(df[cols].to_string(index=False))
    print("=" * 100)


def plot_comparison(df: pd.DataFrame, output_dir: Path, metric: str | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metric:
        candidate_metrics = [metric] if metric in df.columns else []
        if not candidate_metrics:
            print(f"  ⚠ Métrique '{metric}' absente des rapports, graphique ignoré.")
            return
    else:
        preferred = ["box_map50", "mask_map50", "map50", "box_map", "mask_map", "map"]
        candidate_metrics = [m for m in preferred if m in df.columns]
        if not candidate_metrics:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            candidate_metrics = numeric_cols[:4]

    if not candidate_metrics:
        print("  ⚠ Aucune métrique numérique trouvée, graphique ignoré.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.2), 5))
    x = range(len(df))
    width = 0.8 / max(len(candidate_metrics), 1)

    for i, m in enumerate(candidate_metrics):
        values = df[m].astype(float).fillna(0)
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, values, width=width, label=m)

    ax.set_xticks([xi + width * (len(candidate_metrics) - 1) / 2 for xi in x])
    ax.set_xticklabels(df["run"], rotation=30, ha="right")
    ax.set_ylabel("Valeur")
    ax.set_title("Comparaison des runs d'entraînement")
    ax.legend()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "comparison.png"
    fig.savefig(plot_path, dpi=150)
    print(f"  Graphique : {plot_path}")


def main():
    args = parse_args()
    paths = find_reports(args.reports, args.results_dir)

    print(f"\n{len(paths)} rapport(s) trouvé(s) :")
    for p in paths:
        print(f"  - {p}")

    rows = load_reports(paths)
    if not rows:
        sys.exit("❌ Aucun rapport valide n'a pu être chargé.")

    df = pd.DataFrame(rows)
    # Ordonne les colonnes : identifiants d'abord, puis métriques triées
    id_cols = ["run", "model_type", "split"]
    metric_cols = sorted([c for c in df.columns if c not in id_cols + ["fichier"]])
    df = df[id_cols + metric_cols + ["fichier"]]

    print_table(df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Tableau CSV : {csv_path}")

    if not args.no_plot:
        plot_comparison(df, output_dir, args.metric)

    print("✓ Comparaison terminée.")


if __name__ == "__main__":
    main()
