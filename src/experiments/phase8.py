"""Build comparison tables and figures only from completed run artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.chronos import probabilistic_metrics


PHASE4 = {
    "Naive": "naive-h30",
    "SeasonalNaive24": "seasonal_naive_24-h30",
    "SeasonalNaive168": "seasonal_naive_168-h30",
    "AutoETS24": "auto_ets-h30",
    "LightGBM-direct": "lightgbm-h30",
    "MLP": "mlp-L168-per_series-h30",
    "DLinear": "dlinear-L168-per_series-h30",
    "NLinear": "nlinear-L168-per_series-h30",
}
ADVANCED = ("NHITS", "TCN", "PatchTST", "TFT", "DeepAR")
RECURRENT = {
    "lstm_recursive": "LSTM-Recursive",
    "lstm_multioutput": "LSTM-MultiOutput",
    "lstm_seq2seq": "LSTM-Seq2Seq",
}


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def expected_runs(output_dir: Path) -> dict[str, str]:
    selected_path = output_dir / "metrics" / "phase5_selected.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Validation selection missing: {selected_path}")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if set(selected) != set(RECURRENT):
        raise ValueError("Phase 5 selection must contain all LSTM strategies")
    runs = dict(PHASE4)
    for key, name in RECURRENT.items():
        runs[name] = selected[key]["config"]["config_id"]
    for name in ADVANCED:
        runs[name] = f"{name.lower()}-L168-h30-v1"
    runs["Chronos-Bolt"] = "chronos-bolt-L168"
    return runs


def _read_run(output_dir: Path, name: str, config_id: str) -> tuple[dict, Path]:
    directory = output_dir / "predictions" / _slug(name) / config_id
    path = directory / "experiment.json"
    if not path.exists():
        raise FileNotFoundError(f"Required experiment missing: {path}")
    experiment = json.loads(path.read_text(encoding="utf-8"))
    if experiment["model"] != name or experiment["config_id"] != config_id:
        raise ValueError(f"Experiment identity differs from selection: {path}")
    if int(experiment["metrics"]["n_observations"]) != (
        92 * 24 * int(experiment["metrics"]["n_series"])
    ):
        raise ValueError(f"Incomplete 92-origin experiment: {path}")
    return experiment, directory


def _target_fingerprint(directory: Path) -> str:
    columns = pd.read_parquet(directory / "predictions.parquet",
                              columns=["unique_id", "ds", "y_true"])
    if columns.duplicated(["unique_id", "ds"]).any():
        raise ValueError(f"Duplicate client/timestamp predictions: {directory}")
    columns = columns.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(columns, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _family(name: str) -> str:
    if name in {"Naive", "SeasonalNaive24", "SeasonalNaive168", "AutoETS24"}:
        return "Estadístico"
    if name == "LightGBM-direct":
        return "Árboles"
    if name in {"MLP", "DLinear", "NLinear"}:
        return "MLP / lineal"
    if name.startswith("LSTM"):
        return "Recurrente"
    if name == "Chronos-Bolt":
        return "Fundacional"
    if name == "DeepAR":
        return "Probabilístico"
    return "NeuralForecast"


def _strategy(name: str) -> str:
    if name == "LSTM-Recursive":
        return "Autorregresiva"
    if name == "LSTM-Seq2Seq":
        return "Encoder-decoder"
    if name == "LightGBM-direct":
        return "Directa por paso"
    return "Multi-salida"


def _read_probabilistic(output_dir: Path, name: str,
                        config_id: str) -> dict:
    if name == "DeepAR":
        path = (output_dir / "predictions" / "deepar" /
                config_id / "quantiles.parquet")
    elif name == "Chronos-Bolt":
        path = output_dir / "predictions" / "chronos_bolt_quantiles.parquet"
    else:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Quantile artifact missing: {path}")
    return probabilistic_metrics(pd.read_parquet(
        path, columns=["y_true", "p10", "p90"]
    ))


def build_table(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    runs = expected_runs(output_dir)
    rows = []
    directories = {}
    client_count = None
    target_fingerprint = None
    for name, config_id in runs.items():
        experiment, directory = _read_run(output_dir, name, config_id)
        directories[name] = directory
        metrics = experiment["metrics"]
        if client_count is None:
            client_count = int(metrics["n_series"])
        if int(metrics["n_series"]) != client_count:
            raise ValueError(f"Client count differs for {name}")
        fingerprint = _target_fingerprint(directory)
        if target_fingerprint is None:
            target_fingerprint = fingerprint
        elif fingerprint != target_fingerprint:
            raise ValueError(f"Client/timestamp targets differ for {name}")
        christmas = pd.read_csv(directory / "metrics_christmas_week.csv")
        week_mase = float(christmas["mase"].mean()) if len(christmas) else None
        probability = _read_probabilistic(output_dir, name, config_id)
        rows.append({
            "modelo": name, "familia": _family(name),
            "L": experiment["L"], "H": experiment["H"],
            "estrategia": _strategy(name),
            "covariables": name in {"MLP", "LSTM-MultiOutput",
                                    "LSTM-Seq2Seq", "NHITS", "TCN",
                                    "TFT", "DeepAR", "LightGBM-direct"},
            "parametros_totales": experiment["params_total"],
            "parametros_entrenables": experiment["params_trainable"],
            "entrenamiento_s": experiment["train_time_s"],
            "inferencia_s": experiment["inference_time_s"],
            "hardware": experiment["hardware"]["device"],
            "MAE_kWh": metrics["mae"], "RMSE_kWh": metrics["rmse"],
            "WAPE_pct": metrics["wape_pct"], "MASE_H24": metrics["mase"],
            "MASE_H168": None,
            "MASE_22_28_dic": week_mase,
            "pinball_P10": probability.get("pinball_p10"),
            "pinball_P90": probability.get("pinball_p90"),
            "cobertura_P10_P90": probability.get("coverage_p10_p90"),
            "config_id": config_id,
        })
    table = pd.DataFrame(rows).sort_values("MASE_H24", na_position="last")
    destination = output_dir / "metrics" / "comparison_final.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return table, directories


def make_figures(table: pd.DataFrame, directories: dict[str, Path],
                 output_dir: Path) -> None:
    figure_dir = output_dir / "figures" / "final"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trainable = table.loc[table["entrenamiento_s"].gt(0)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(trainable["entrenamiento_s"], trainable["MASE_H24"])
    for row in trainable.itertuples():
        ax.annotate(row.modelo, (row.entrenamiento_s, row.MASE_H24),
                    fontsize=7)
    pareto = trainable.sort_values("entrenamiento_s")
    frontier = pareto.loc[pareto["MASE_H24"].cummin().eq(pareto["MASE_H24"])]
    ax.plot(frontier["entrenamiento_s"], frontier["MASE_H24"],
            color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo de entrenamiento (s, escala log)")
    ax.set_ylabel("MASE de prueba, H=24")
    ax.set_title("Error frente a costo de entrenamiento")
    fig.tight_layout()
    fig.savefig(figure_dir / "pareto_mase_tiempo.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, directory in directories.items():
        series = pd.read_csv(directory / "metrics_by_horizon.csv")
        ax.plot(series["horizon_step"], series["mae"], label=name)
    ax.set_xlabel("Paso del horizonte (horas)")
    ax.set_ylabel("MAE (kWh)")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "mae_por_paso.png", dpi=180)
    plt.close(fig)

    history_path = output_dir / "metrics" / "phase7_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Required history experiment missing: {history_path}")
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in history.groupby("model", sort=True):
        ax.plot(group["history_months"], group["test_mase"],
                marker="o", label=name)
    chronos = table.loc[table["modelo"].eq("Chronos-Bolt"), "MASE_H24"].iloc[0]
    ax.axhline(chronos, color="black", linestyle="--", label="Chronos-Bolt")
    ax.set_xlabel("Meses de historia de entrenamiento")
    ax.set_ylabel("MASE de prueba, H=24")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "mase_por_historia.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("LSTM-Recursive", "LSTM-MultiOutput", "LSTM-Seq2Seq"):
        series = pd.read_csv(directories[name] / "metrics_by_horizon.csv")
        ax.plot(series["horizon_step"], series["mae"], label=name)
    ax.set_xlabel("Paso del horizonte (horas)")
    ax.set_ylabel("MAE (kWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "estrategias_lstm.png", dpi=180)
    plt.close(fig)

    validation_path = output_dir / "metrics" / "phase5_validation.csv"
    if not validation_path.exists():
        raise FileNotFoundError(f"Training curves missing: {validation_path}")
    validation = pd.read_csv(validation_path)
    curves = validation.loc[validation["model"].eq("lstm_multioutput")]
    if len(curves) < 3:
        raise ValueError("At least three LSTM iterations are needed for curves")
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in curves.itertuples():
        log = pd.read_csv(row.training_log)
        ax.plot(log["epoch"], log["validation_loss"],
                label=f"L={row.lookback} validación")
        ax.plot(log["epoch"], log["train_loss"], linestyle="--",
                label=f"L={row.lookback} entrenamiento")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE normalizado")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "curvas_tres_iteraciones.png", dpi=180)
    plt.close(fig)


def write_report_draft(table: pd.DataFrame, output_dir: Path) -> Path:
    """Write factual comparisons and leave interpretation for human review."""
    by_name = table.set_index("modelo")
    winner = table.iloc[0]
    seasonal = by_name.loc["SeasonalNaive24"]
    improvement = 100 * (
        seasonal["MASE_H24"] - winner["MASE_H24"]
    ) / seasonal["MASE_H24"]
    recurrent = table.loc[table["modelo"].str.startswith("LSTM")].iloc[0]
    dlinear = by_name.loc["DLinear"]
    patchtst = by_name.loc["PatchTST"]
    chronos = by_name.loc["Chronos-Bolt"]
    seeds = pd.read_csv(output_dir / "metrics" / "phase7_seeds.csv")
    local = pd.read_csv(output_dir / "metrics" / "phase7_local_global.csv")
    monthly = pd.read_csv(output_dir / "metrics" / "phase7_monthly.csv")
    history = pd.read_csv(output_dir / "metrics" / "phase7_history.csv")
    columns = ["modelo", "familia", "L", "MAE_kWh", "RMSE_kWh",
               "WAPE_pct", "MASE_H24", "entrenamiento_s", "inferencia_s"]
    lines = [
        "# Laboratorio 8: series de tiempo y forecasting",
        "",
        "**Borrador generado desde experimentos; revisar antes de convertir a PDF.**",
        "",
        "## Método",
        "",
        "Panel horario en kWh; pronóstico day-ahead de 24 horas, 92 orígenes "
        "diarios de octubre a diciembre de 2014. Splits cronológicos y MASE "
        "estacional de 24 horas. Consultar README y auditoría de datos para "
        "unidades, horario local y fugas.",
        "",
        "## Comparación en prueba",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(
            "N/A" if pd.isna(row[column]) else
            (f"{row[column]:.4f}" if isinstance(row[column], (float, np.floating))
             else str(row[column]))
            for column in columns
        ) + " |")
    lines.extend([
        "", "## Hallazgos cuantitativos", "",
        f"- Menor MASE H=24: {winner['modelo']} ({winner['MASE_H24']:.4f}); "
        f"diferencia frente a SeasonalNaive24: {improvement:.2f}%.",
        f"- DLinear: MASE {dlinear['MASE_H24']:.4f}; "
        f"PatchTST: {patchtst['MASE_H24']:.4f}. La comparación H=168 "
        "queda N/A si no se ejecutó ese horizonte.",
        f"- Mejor estrategia LSTM en esta tabla: {recurrent['modelo']} "
        f"(MASE {recurrent['MASE_H24']:.4f}). Comparar acumulación de error "
        "en `outputs/figures/final/estrategias_lstm.png`.",
        f"- Chronos-Bolt zero-shot: MASE {chronos['MASE_H24']:.4f}; "
        f"tiempo de inferencia {chronos['inferencia_s']:.1f} s.",
        f"- Sensibilidad a semillas: media MASE {seeds['mean_mase'].iloc[0]:.4f}, "
        f"desviación estándar muestral {seeds['sd_mase'].iloc[0]:.4f}.",
        f"- Modelo global supera a los locales en "
        f"{int((local['global_mase'] < local['local_mase']).sum())}/10 clientes.",
        "", "## Experimentos complementarios", "",
        f"- Historia: {len(history)} configuraciones predeclaradas; "
        "ver `outputs/figures/final/mase_por_historia.png`.",
        f"- Ajuste mensual: {len(monthly)} modelos; "
        "ver `outputs/metrics/phase7_monthly.csv`.",
        "- Cuantiles de DeepAR y Chronos: ver columnas pinball y cobertura "
        "en `outputs/metrics/comparison_final.csv`.",
        "- Analizar con las figuras y predicciones las diferencias de Navidad, "
        "Año Nuevo, costo de mantenimiento, normalización y efectos por paso.",
        "", "## Referencias y repositorio", "",
        "- UCI ElectricityLoadDiagrams20112014: "
        "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
        "- Documentación de PyTorch, Nixtla y Chronos: consultar las notas "
        "de investigación del proyecto.",
        "- Enlace Git del equipo: **AGREGAR URL REAL ANTES DE ENTREGAR**.",
        "",
    ])
    destination = Path("report") / "informe_borrador.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    table, directories = build_table(args.output_dir)
    make_figures(table, directories, args.output_dir)
    draft = write_report_draft(table, args.output_dir)
    print(f"Borrador del informe: {draft}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
