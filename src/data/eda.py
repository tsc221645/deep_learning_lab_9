"""Generate exploratory summaries and figures from the hourly panel."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.calendars import calendar_features, portugal_holidays
from src.data.panel import load_panel_matrix
from src.data.splits import (
    DEFAULT_SPLITS,
    daily_backtest_origins,
    temporal_split_origins,
)


def run_eda(
    panel_path: Path,
    client_ids_path: Path,
    output_dir: Path = Path("outputs"),
) -> dict:
    """Create reproducible EDA charts, client scales, and window counts."""
    panel = load_panel_matrix(panel_path, client_ids_path)
    figure_dir = output_dir / "figures" / "eda"
    metrics_dir = output_dir / "metrics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    timestamps = panel.timestamps
    average_load = panel.values.mean(axis=0, dtype=np.float64)
    hourly = pd.DataFrame(
        {
            "ds": timestamps,
            "mean_kwh": average_load,
            "hour": timestamps.hour,
            "weekday": timestamps.dayofweek,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(
        hourly.groupby("hour")["mean_kwh"].mean(),
        marker="o",
        linewidth=1.5,
    )
    axes[0].set(
        title="Consumo horario promedio por cliente",
        xlabel="Hora local",
        ylabel="kWh promedio por cliente",
    )
    for weekday in range(7):
        profile = (
            hourly.loc[hourly["weekday"] == weekday]
            .groupby("hour")["mean_kwh"]
            .mean()
        )
        axes[1].plot(profile.index, profile.values, label=weekday)
    axes[1].set(
        title="Perfil por día de semana",
        xlabel="Hora local",
        ylabel="kWh promedio por cliente",
    )
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].legend(
        [("Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom")[day] for day in range(7)],
        ncol=4,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "hourly_weekly_profiles.png", dpi=160)
    plt.close(fig)

    calendar = calendar_features(timestamps)
    holiday_hours = pd.DataFrame(
        {
            "hour": timestamps.hour,
            "holiday": calendar["is_holiday"].to_numpy(dtype=bool),
            "mean_kwh": average_load,
        }
    )
    holiday_profile = holiday_hours.groupby(["holiday", "hour"])["mean_kwh"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for is_holiday, label in ((False, "Día regular"), (True, "Feriado nacional")):
        if is_holiday in holiday_profile.index.get_level_values("holiday"):
            profile = holiday_profile.loc[is_holiday]
            ax.plot(profile.index, profile.values, marker=".", label=label)
    ax.set(
        title="Perfil horario en feriados de Portugal",
        xlabel="Hora local",
        ylabel="kWh promedio por cliente",
    )
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "portugal_holidays.png", dpi=160)
    plt.close(fig)
    holiday_map = portugal_holidays(timestamps.year)
    holiday_table = pd.DataFrame(
        [
            {"date": date.isoformat(), "holiday": name}
            for date, name in sorted(holiday_map.items())
            if timestamps[0].date() <= date <= timestamps[-1].date()
        ]
    )
    holiday_table.to_csv(metrics_dir / "portugal_holidays.csv", index=False)

    train_start, train_end = DEFAULT_SPLITS["train"]
    train_mask = (timestamps >= train_start) & (timestamps < train_end)
    if not train_mask.any():
        raise ValueError("No training observations are available for scale summaries")
    client_means = panel.values[:, train_mask].mean(axis=1, dtype=np.float64)
    scales = pd.DataFrame(
        {
            "unique_id": panel.client_ids,
            "mean_hourly_kwh_train": client_means,
        }
    )
    scales.to_csv(metrics_dir / "client_scales.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        np.log10(1 + client_means),
        bins=30,
        color="#4472C4",
        edgecolor="white",
    )
    ax.set(
        title="Escalas promedio por cliente (log1p)",
        xlabel="log10(1 + kWh por hora promedio en entrenamiento)",
        ylabel="Cantidad de clientes",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "client_scales.png", dpi=160)
    plt.close(fig)

    daily_average = pd.Series(average_load, index=timestamps).groupby(
        timestamps.normalize()
    ).sum()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for year in sorted(set(timestamps.year)):
        dates = pd.date_range(f"{year}-12-22", f"{year + 1}-01-02", freq="D")
        observed = daily_average.reindex(dates)
        ax.plot(
            np.arange(len(dates)),
            observed.to_numpy(),
            marker="o",
            label=str(year),
        )
    tick_labels = pd.date_range("2012-12-22", "2013-01-02", freq="D").strftime(
        "%m-%d"
    )
    ax.set(
        title="Consumo alrededor de Navidad y Año Nuevo",
        xlabel="Día",
        ylabel="kWh diarios promedio por cliente",
    )
    ax.set_xticks(range(len(tick_labels)), tick_labels, rotation=45)
    ax.axvspan(0, 6, color="#F4B183", alpha=0.15, label="22 al 28 de diciembre")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "christmas_new_year.png", dpi=160)
    plt.close(fig)

    origins = temporal_split_origins(
        timestamps, lookback=168, horizon=24, stride=1
    )
    daily_test_origins = daily_backtest_origins(timestamps)
    window_counts = pd.DataFrame(
        [
            {
                "split": name,
                "origins_per_client": len(indices),
                "windows_all_clients": len(indices) * len(panel.client_ids),
                "daily_backtest_origins": (
                    len(daily_test_origins) if name == "test" else 0
                ),
            }
            for name, indices in origins.items()
        ]
    )
    window_counts.to_csv(metrics_dir / "window_counts.csv", index=False)

    summary = {
        "client_count": len(panel.client_ids),
        "hourly_rows_per_client": len(timestamps),
        "first_timestamp": timestamps[0].isoformat(sep=" "),
        "last_timestamp": timestamps[-1].isoformat(sep=" "),
        "train_mean_hourly_kwh_min": float(client_means.min()),
        "train_mean_hourly_kwh_median": float(np.median(client_means)),
        "train_mean_hourly_kwh_max": float(client_means.max()),
        "portugal_public_holiday_dates": len(holiday_table),
        "calendar_features_cyclic": int(calendar_features(timestamps[:1]).shape[1]),
        "calendar_features_onehot": int(
            calendar_features(timestamps[:1], encoding="onehot").shape[1]
        ),
        "window_counts": window_counts.to_dict(orient="records"),
        "normalization_note": (
            "Compare global and per-series scaling; fit statistics only on "
            "the training interval. Window scaling uses its history only."
        ),
        "eda_scope": (
            "Descriptive plots use the available panel and are not used to tune "
            "models on the test period."
        ),
    }
    (metrics_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path, default=Path("data/processed/hourly.parquet")
    )
    parser.add_argument(
        "--client-ids", type=Path, default=Path("data/processed/client_ids.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    print(
        json.dumps(
            run_eda(args.panel, args.client_ids, args.output_dir),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
