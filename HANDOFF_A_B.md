# Entrega de A a B

Esta entrega completa las fases 1 a 4. La fase 5 puede reutilizar la infraestructura sin cambiar las decisiones de datos ni el protocolo de evaluación.

## Panel y calendario

- Panel horario en kWh: `data/processed/hourly.parquet`.
- IDs congelados: `data/processed/client_ids.csv`.
- Carga eficiente: `load_panel_matrix` en `src/data/panel.py`.
- Splits cronológicos: entrenamiento hasta `2014-07-01`, validación hasta `2014-10-01`, prueba hasta `2015-01-01`, definidos en `src/data/splits.py`.
- La hora local portuguesa se mantiene sin zona horaria. No se convierte a UTC.
- Las covariables conocidas se crean con `calendar_features`; no se deben mezclar con targets futuros.

## Modelos e interfaces

Los adaptadores implementan `fit(train, val, config)` cuando requieren ajuste y `predict(history, future_calendar, horizon, origin)`, con salida `(clientes, horizonte)` en kWh. El evaluador verifica forma, finitud, alineación temporal y ausencia de targets en el calendario futuro.

Los modelos de esta entrega están en `src/models/`: cuatro baselines StatsForecast, LightGBM directo a 24 pasos, MLP multi-salida, DLinear y NLinear. La alternativa estadística seleccionada es AutoETS(A,A,A) con estacionalidad diaria. Los modelos lineales y el MLP usan `Trainer` y `TemporalScaler`; el estado de escalado se ajusta solo con entrenamiento.

Los modelos PyTorch requieren CUDA y no hacen fallback a CPU. La instalación fijada usa PyTorch `cu128`.

## Evaluación y ejecución

El backtest y las métricas están en `src/evaluation/backtest.py` y `src/evaluation/metrics.py`. Se usa un origen diario de medianoche para cada día de prueba, un horizonte de 24 y MASE con estacionalidad 24. Los targets se leen después de que el adaptador genera el pronóstico.

```powershell
python -m pytest -q
python -m src.experiments.phase4
```

Usa `python -m src.experiments.phase4 --smoke` solo para una comprobación rápida con un origen y hasta ocho clientes. El runner completo evalúa 92 orígenes y guarda pronósticos, métricas, configuraciones, logs y checkpoints en `outputs/`. Las predicciones no se versionan por su tamaño; se regeneran con el comando anterior. Consulta `docs/fase3_evaluacion.md` y `docs/fase4_modelos.md` antes de cambiar contratos o decisiones metodológicas.

## Decisiones que no deben alterarse al integrar fase 5

- Mantener IDs, unidad kWh, zona horaria local, splits y horizonte H=24.
- Usar el evaluador común y no alimentar el modelo con consumos posteriores al origen.
- Ajustar hiperparámetros y early stopping solo con validación. El test no interviene en la selección.
- Mantener métricas en escala original y el mismo tratamiento de MAPE y MASE.
- Añadir modelos como adaptadores; no duplicar ni reconstruir el pipeline de datos.
