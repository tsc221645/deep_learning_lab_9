# Informe de estado del laboratorio

**Avance de validación. No es el informe final ni contiene resultados comparativos de prueba.**

- Fecha: 2026-11-01
- Commit base de esta revisión: `3b1df1e`
- Repositorio: https://github.com/tsc221645/deep_learning_lab_9
- Pruebas: 46 aprobadas.
- Entorno disponible: NVIDIA GeForce RTX 3050, PyTorch 2.11.0+cu128.
- Dispositivo registrado en smoke: 12 corridas CUDA y 5 CPU.
- Naive, SeasonalNaive, AutoETS y LightGBM usan implementaciones CPU en esta configuración, sin ruta CUDA. No se debe afirmar cumplimiento completo del requisito de GPU.

## Smoke de integración

Cada corrida usa ocho clientes, un origen y 192 pronósticos. Estos datos solo validan la interfaz y no estiman el rendimiento de los modelos.

| Modelo | Clientes | Orígenes | Pronósticos | Dispositivo registrado |
|---|---:|---:|---:|---|
| AutoETS24 | 8 | 1 | 192 | cpu |
| Chronos-Bolt | 8 | 1 | 192 | cuda |
| DLinear | 8 | 1 | 192 | cuda |
| DeepAR | 8 | 1 | 192 | cuda |
| LSTM-MultiOutput | 8 | 1 | 192 | cuda |
| LSTM-Recursive | 8 | 1 | 192 | cuda |
| LSTM-Seq2Seq | 8 | 1 | 192 | cuda |
| LightGBM-direct | 8 | 1 | 192 | cpu |
| MLP | 8 | 1 | 192 | cuda |
| NHITS | 8 | 1 | 192 | cuda |
| NLinear | 8 | 1 | 192 | cuda |
| Naive | 8 | 1 | 192 | cpu |
| PatchTST | 8 | 1 | 192 | cuda |
| SeasonalNaive168 | 8 | 1 | 192 | cpu |
| SeasonalNaive24 | 8 | 1 | 192 | cpu |
| TCN | 8 | 1 | 192 | cuda |
| TFT | 8 | 1 | 192 | cuda |

## Pendiente antes de la entrega

- Ejecutar los experimentos completos con los 370 clientes y 92 orígenes, sin usar el conjunto de prueba para seleccionar hiperparámetros.
- Completar sensibilidad de historia, horizonte, semillas, comparación local/global y reajustes mensuales.
- Verificar cuantiles y artefactos, analizar los resultados y completar la interpretación.
- Generar y revisar el PDF final de cinco páginas como máximo.

Artefactos finales que aún faltan:

- `outputs/metrics/phase4_summary.csv`
- `outputs/metrics/phase5_selected.json`
- `outputs/metrics/phase5_test.csv`
- `outputs/metrics/phase6_summary.csv`
- `outputs/metrics/phase7_history.csv`
- `outputs/metrics/phase7_monthly.csv`
- `outputs/metrics/phase7_local_global.csv`
- `outputs/metrics/phase7_seeds.csv`
- `outputs/metrics/comparison_final.csv`

Los smoke no se deben citar como métricas finales ni como evidencia de cuál modelo gana.
