"""Create an explicitly exploratory report from the existing smoke artifacts.

This is separate from phase8.py, which requires the complete backtest.
No model is trained by this module.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from src.experiments.report_status import EXPECTED_MODELS


REQUIRED_COLUMNS = {
    "unique_id", "origin", "ds", "horizon_step", "y_true", "y_hat"
}
TARGET_COLUMNS = ["unique_id", "origin", "ds", "horizon_step", "y_true"]


def collect_comparison(outputs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read 17 runs and require the same eight clients and actual targets."""
    roots = (outputs / "predictions", outputs / "smoke_b" / "predictions")
    paths = sorted(path for root in roots for path in root.glob("*/*/experiment.json"))
    found: dict[str, Path] = {}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("model") not in EXPECTED_MODELS or not str(
            record.get("config_id", "")
        ).endswith("-smoke"):
            continue
        name = record["model"]
        if name in found:
            raise ValueError(f"Multiple smoke runs for {name}: {found[name]}, {path}")
        found[name] = path
    missing = EXPECTED_MODELS - found.keys()
    if missing:
        raise FileNotFoundError(f"Missing smoke models: {sorted(missing)}")

    reference = None
    rows = []
    horizon_rows = []
    for name, path in sorted(found.items()):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["model"] != name or record["config_id"] != path.parent.name:
            raise ValueError(f"Experiment identity differs from directory: {path}")
        if int(record["H"]) != 24:
            raise ValueError(f"Expected H=24 in {path}")
        predictions_path = path.parent / "predictions.parquet"
        if not predictions_path.is_file():
            raise FileNotFoundError(predictions_path)
        frame = pd.read_parquet(predictions_path)
        missing_columns = REQUIRED_COLUMNS - set(frame.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in {predictions_path}: {sorted(missing_columns)}")
        if len(frame) != 192 or frame["unique_id"].nunique() != 8:
            raise ValueError(f"Expected 8 clients and 192 predictions in {predictions_path}")
        if frame["origin"].nunique() != 1 or frame["horizon_step"].nunique() != 24:
            raise ValueError(f"Expected one origin and 24 horizon steps in {predictions_path}")
        expected_steps = set(range(1, 25))
        steps_by_client = frame.groupby("unique_id")["horizon_step"].agg(
            lambda steps: set(steps) == expected_steps and len(steps) == 24
        )
        if not steps_by_client.all():
            raise ValueError(f"Incomplete 24-hour forecast in {predictions_path}")
        if not np.isfinite(frame[["y_true", "y_hat"]].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite target or prediction in {predictions_path}")
        if frame.duplicated(["unique_id", "origin", "ds"]).any():
            raise ValueError(f"Duplicate prediction keys in {predictions_path}")
        targets = frame[TARGET_COLUMNS].sort_values(
            ["unique_id", "origin", "ds"], kind="stable"
        ).reset_index(drop=True)
        if reference is None:
            reference = targets
        else:
            try:
                pd.testing.assert_frame_equal(reference, targets, check_exact=True)
            except AssertionError as exc:
                raise ValueError(f"Targets or dates differ for {name}") from exc

        actual = frame["y_true"].to_numpy(dtype=float)
        error = np.abs(actual - frame["y_hat"].to_numpy(dtype=float))
        denominator = np.abs(actual).sum()
        if denominator <= 0:
            raise ValueError("WAPE is undefined because all targets are zero")
        rows.append({
            "modelo": name,
            "config_id": record["config_id"],
            "historia_meses": record["history_months"],
            "clientes": 8,
            "origenes": 1,
            "pronosticos": len(frame),
            "MAE_kWh": float(error.mean()),
            "RMSE_kWh": float(np.sqrt(np.mean(np.square(error)))),
            "WAPE_pct": float(100 * error.sum() / denominator),
            "entrenamiento_s": record["train_time_s"],
            "inferencia_s": record["inference_time_s"],
            "dispositivo": record["hardware"]["device"],
        })
        for step, group in frame.groupby("horizon_step", sort=True):
            horizon_rows.append({
                "modelo": name, "paso": int(step),
                "MAE_kWh": float(np.abs(group["y_true"] - group["y_hat"]).mean()),
            })

    comparison = pd.DataFrame(rows).sort_values("MAE_kWh").reset_index(drop=True)
    by_horizon = pd.DataFrame(horizon_rows)
    return comparison, by_horizon


def write_report(comparison: pd.DataFrame, by_horizon: pd.DataFrame,
                 outputs: Path, report_dir: Path, repo_url: str) -> None:
    metrics_dir = outputs / "metrics"
    figures_dir = outputs / "figures" / "smoke"
    for folder in (metrics_dir, figures_dir, report_dir):
        folder.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_dir / "comparison_smoke.csv", index=False)
    by_horizon.to_csv(metrics_dir / "comparison_smoke_by_horizon.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(comparison["modelo"], comparison["MAE_kWh"], color="#4776a8")
    ax.invert_yaxis()
    ax.set_xlabel("MAE (kWh)")
    ax.set_title("SMOKE exploratorio: 8 clientes, 1 origen")
    fig.tight_layout()
    fig.savefig(figures_dir / "mae_smoke.png", dpi=160)
    plt.close(fig)

    columns = ["modelo", "historia_meses", "MAE_kWh", "RMSE_kWh", "WAPE_pct"]
    lines = [
        "# Fase 8: comparación exploratoria de smoke",
        "",
        "**Resultado parcial. No equivale a la evaluación final del laboratorio.**",
        "",
        "Los 17 modelos tienen predicciones guardadas para los mismos 8 clientes, "
        "el mismo origen de prueba y 24 horas (192 pronósticos por modelo). "
        "El programa verifica identidad de clientes, fechas y valores reales "
        "antes de generar la tabla. No entrena ni ejecuta modelos.",
        "",
        "La fase 4 smoke usó 30 meses de historia y los modelos de B usaron 3. "
        "Por eso se recalculan MAE, RMSE y WAPE directamente desde las predicciones; "
        "no se combinan los MASE guardados, que usan escalas distintas. "
        "La diferencia de historia también limita la interpretación de esta comparación.",
        "",
        "## Métricas del único origen",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.modelo} | {row.historia_meses} | {row.MAE_kWh:.3f} | "
            f"{row.RMSE_kWh:.3f} | {row.WAPE_pct:.2f} |"
        )
    lines.extend([
        "",
        "La tabla está ordenada por MAE solo para facilitar la lectura. Un origen "
        "no permite declarar un modelo ganador ni medir estabilidad, Navidad o Año Nuevo.",
        "",
        "## Artefactos",
        "",
        "- `outputs/metrics/comparison_smoke.csv`: métricas y metadatos de los 17 modelos.",
        "- `outputs/metrics/comparison_smoke_by_horizon.csv`: MAE por paso.",
        "- `outputs/figures/smoke/mae_smoke.png`: gráfico de MAE.",
        "- `report/informe_smoke.pdf`: copia breve para revisión.",
        "",
        "Para completar la fase 8 original hacen falta las corridas de 370 clientes "
        "y 92 orígenes, la selección de validación de fase 5 y los experimentos "
        "de fase 7. Este informe no responde esas preguntas.",
        "",
        f"Repositorio: {repo_url}",
        "",
    ])
    (report_dir / "informe_smoke.md").write_text("\n".join(lines), encoding="utf-8")

    with PdfPages(report_dir / "informe_smoke.pdf") as pdf:
        page, ax = plt.subplots(figsize=(8.3, 11.7))
        ax.axis("off")
        ax.text(0.02, 0.98, "Fase 8: smoke exploratorio", fontsize=17,
                weight="bold", va="top", transform=ax.transAxes)
        ax.text(0.02, 0.935,
                "Resultado parcial: 8 clientes, 1 origen, 24 horas. No es evaluación final.",
                fontsize=9, va="top", transform=ax.transAxes)
        displayed = comparison[["modelo", "MAE_kWh", "RMSE_kWh", "WAPE_pct"]].copy()
        for column in ("MAE_kWh", "RMSE_kWh", "WAPE_pct"):
            displayed[column] = displayed[column].map(lambda value: f"{value:.2f}")
        table = ax.table(cellText=displayed.values,
                         colLabels=["Modelo", "MAE", "RMSE", "WAPE %"],
                         cellLoc="left", colLoc="left",
                         colWidths=[0.46, 0.17, 0.17, 0.18],
                         bbox=[0.02, 0.20, 0.96, 0.70])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax.text(0.02, 0.14,
                "Fase 4: 30 meses de historia. Modelos B: 3 meses. "
                "MASE no comparable entre ambas corridas.",
                fontsize=8, transform=ax.transAxes, wrap=True)
        ax.text(0.02, 0.08,
                "Un solo origen no sustenta una conclusión sobre el mejor modelo.",
                fontsize=8, transform=ax.transAxes)
        ax.text(0.02, 0.03, repo_url, fontsize=7, transform=ax.transAxes)
        pdf.savefig(page)
        plt.close(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    parser.add_argument("--repo-url", default="https://github.com/tsc221645/deep_learning_lab_9")
    args = parser.parse_args()
    comparison, by_horizon = collect_comparison(args.output_dir)
    write_report(comparison, by_horizon, args.output_dir, args.report_dir, args.repo_url)
    print("Smoke comparison: outputs/metrics/comparison_smoke.csv")
    print("Exploratory report: report/informe_smoke.md and report/informe_smoke.pdf")


if __name__ == "__main__":
    main()
