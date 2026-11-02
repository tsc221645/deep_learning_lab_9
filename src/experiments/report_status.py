"""Generate a factual, explicitly non-final PDF status report from smoke runs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

EXPECTED_MODELS = {
    "Naive", "SeasonalNaive24", "SeasonalNaive168", "AutoETS24",
    "LightGBM-direct", "MLP", "DLinear", "NLinear",
    "LSTM-Recursive", "LSTM-MultiOutput", "LSTM-Seq2Seq",
    "NHITS", "TCN", "PatchTST", "TFT", "DeepAR", "Chronos-Bolt",
}
EXPECTED_SMOKE_ROWS = 8 * 24
MISSING_FINAL_ARTIFACTS = (
    "outputs/metrics/phase4_summary.csv",
    "outputs/metrics/phase5_selected.json",
    "outputs/metrics/phase5_test.csv",
    "outputs/metrics/phase6_summary.csv",
    "outputs/metrics/phase7_history.csv",
    "outputs/metrics/phase7_monthly.csv",
    "outputs/metrics/phase7_local_global.csv",
    "outputs/metrics/phase7_seeds.csv",
    "outputs/metrics/comparison_final.csv",
)


def collect_smoke_runs(
    project_root: Path, output_dir: Path | None = None
) -> pd.DataFrame:
    """Validate all expected phase 4 and B smoke experiment records."""
    outputs = output_dir or project_root / "outputs"
    if not outputs.is_absolute():
        outputs = project_root / outputs
    roots = (
        outputs / "predictions",
        outputs / "smoke_b" / "predictions",
    )
    experiments = []
    for root in roots:
        if root.exists():
            experiments.extend(root.glob("*/*/experiment.json"))

    by_name = {}
    for path in experiments:
        record = json.loads(path.read_text(encoding="utf-8"))
        name = record.get("model")
        config_id = record.get("config_id", "")
        if name not in EXPECTED_MODELS or not config_id.endswith("-smoke"):
            continue
        metrics = record["metrics"]
        rows = int(metrics["n_observations"])
        clients = int(metrics["n_series"])
        if clients != 8 or rows != EXPECTED_SMOKE_ROWS:
            raise ValueError(f"Unexpected smoke dimensions in {path}")
        by_name[name] = {
            "model": name,
            "config_id": config_id,
            "clients": clients,
            "origins": rows // (clients * 24),
            "predictions": rows,
            "device": record["hardware"]["device"],
        }

    missing = EXPECTED_MODELS.difference(by_name)
    if missing:
        raise FileNotFoundError(
            f"Smoke artifacts are missing for: {sorted(missing)}"
        )
    return pd.DataFrame(by_name.values()).sort_values("model").reset_index(drop=True)


def _repository_url(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    remote = result.stdout.strip()
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.removeprefix("git@github.com:")
    return remote.removesuffix(".git")


def create_status_report(
    project_root: Path,
    output_dir: Path,
    *,
    tests_passed: int,
) -> tuple[Path, Path]:
    if tests_passed < 1:
        raise ValueError("tests_passed must be positive")
    runs = collect_smoke_runs(project_root, output_dir)
    repository_url = _repository_url(project_root)
    missing = [
        path for path in MISSING_FINAL_ARTIFACTS
        if not (project_root / path).exists()
    ]
    report_dir = project_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "informe_estado.md"
    pdf_path = report_dir / "informe_estado.pdf"
    date = datetime.now().astimezone().strftime("%Y-%m-%d")
    commit = _git_value(project_root, "rev-parse", "--short", "HEAD")
    markdown_lines = [
        "# Informe de estado del laboratorio",
        "",
        "**Avance de validación. No es el informe final ni contiene resultados comparativos de prueba.**",
        "",
        f"- Fecha: {date}",
        f"- Commit base de esta revisión: `{commit}`",
        f"- Repositorio: {repository_url}",
        f"- Pruebas: {tests_passed} aprobadas.",
        "- Entorno disponible: NVIDIA GeForce RTX 3050, PyTorch 2.11.0+cu128.",
        (
            f"- Dispositivo registrado en smoke: "
            f"{int(runs['device'].eq('cuda').sum())} corridas CUDA y "
            f"{int(runs['device'].eq('cpu').sum())} CPU."
        ),
        (
            "- Naive, SeasonalNaive, AutoETS y LightGBM usan implementaciones "
            "CPU en esta configuración, sin ruta CUDA. No se debe afirmar "
            "cumplimiento completo del requisito de GPU."
        ),
        "",
        "## Smoke de integración",
        "",
        "Cada corrida usa ocho clientes, un origen y 192 pronósticos. Estos datos solo validan la interfaz y no estiman el rendimiento de los modelos.",
        "",
        "| Modelo | Clientes | Orígenes | Pronósticos | Dispositivo registrado |",
        "|---|---:|---:|---:|---|",
    ]
    for row in runs.itertuples():
        markdown_lines.append(
            f"| {row.model} | {row.clients} | {row.origins} | "
            f"{row.predictions} | {row.device} |"
        )
    markdown_lines.extend(
        [
            "",
            "## Pendiente antes de la entrega",
            "",
            "- Ejecutar los experimentos completos con los 370 clientes y 92 orígenes, sin usar el conjunto de prueba para seleccionar hiperparámetros.",
            "- Completar sensibilidad de historia, horizonte, semillas, comparación local/global y reajustes mensuales.",
            "- Verificar cuantiles y artefactos, analizar los resultados y completar la interpretación.",
            "- Generar y revisar el PDF final de cinco páginas como máximo.",
            "",
            "Artefactos finales que aún faltan:",
            "",
        ]
    )
    markdown_lines.extend(f"- `{path}`" for path in missing)
    markdown_lines.extend(
        [
            "",
            "Los smoke no se deben citar como métricas finales ni como evidencia de cuál modelo gana.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    _write_pdf(
        pdf_path, runs, repository_url, date, tests_passed, missing, commit
    )
    return markdown_path, pdf_path


def _write_pdf(
    path: Path,
    runs: pd.DataFrame,
    repository_url: str,
    date: str,
    tests_passed: int,
    missing: list[str],
    commit: str,
) -> None:
    cuda_runs = int(runs["device"].eq("cuda").sum())
    cpu_runs = int(runs["device"].eq("cpu").sum())
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=15,
        leading=18,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        spaceAfter=4,
    )
    small = ParagraphStyle(
        "ReportSmall",
        parent=body,
        fontSize=6.5,
        leading=8,
        spaceAfter=0,
    )
    story = [
        Paragraph("Informe de estado del laboratorio", title),
        Paragraph(
            "<b>AVANCE, NO INFORME FINAL.</b> Los smoke validan integración, "
            "no comparan el rendimiento de los modelos.",
            body,
        ),
        Paragraph(
            f"Fecha: {date}. Commit base: {commit}. Pruebas: {tests_passed} "
            "aprobadas. Entorno disponible: NVIDIA GeForce RTX 3050, "
            "PyTorch 2.11.0+cu128.",
            body,
        ),
        Paragraph(
            f"Los metadatos registran {cuda_runs} smoke en CUDA y {cpu_runs} "
            "en CPU. Naive, SeasonalNaive, AutoETS y LightGBM usan "
            "implementaciones CPU en esta configuración, sin ruta CUDA. No "
            "se debe afirmar cumplimiento total del requisito de GPU.",
            body,
        ),
        Paragraph(
            "Cada experimento de la tabla usa ocho clientes, un origen y "
            "192 pronósticos. Las corridas completas con 370 clientes y "
            "92 orígenes aún no se han ejecutado.",
            body,
        ),
        Spacer(1, 2 * mm),
    ]
    rows = [
        ["Modelo", "Clientes", "Orígenes", "Pronósticos", "Dispositivo"]
    ]
    rows.extend(
        [
            [
                Paragraph(str(row.model), small),
                str(row.clients),
                str(row.origins),
                str(row.predictions),
                str(row.device),
            ]
            for row in runs.itertuples()
        ]
    )
    table = Table(rows, colWidths=[72 * mm, 20 * mm, 20 * mm, 30 * mm, 20 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7),
                ("FONTSIZE", (1, 1), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AAB4BF")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white, colors.HexColor("#F0F3F6")
                ]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 4 * mm),
            Paragraph("Pendiente antes de entregar", styles["Heading2"]),
            Paragraph(
                "Ejecutar pruebas completas en los 370 clientes y 92 orígenes; "
                "terminar sensibilidad, semillas, comparación local/global, "
                "reajustes mensuales y cuantiles; revisar conclusiones y "
                "generar el PDF final de cinco páginas como máximo.",
                body,
            ),
            Paragraph(
                "Artefactos de resultados finales ausentes: "
                + ", ".join(missing),
                body,
            ),
            Paragraph(
                "No usar las métricas de smoke para elegir un ganador ni "
                "presentarlas como resultados de prueba.",
                body,
            ),
            Paragraph(
                f"Repositorio: <link href='{repository_url}'>{repository_url}</link>",
                body,
            ),
        ]
    )
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="Informe de estado del laboratorio",
        author="Equipo del laboratorio",
    )
    document.build(story)


def _git_value(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    markdown, pdf = create_status_report(
        root,
        args.output_dir,
        tests_passed=args.tests_passed,
    )
    print(f"Informe de estado: {markdown}")
    print(f"PDF de estado: {pdf}")


if __name__ == "__main__":
    main()
