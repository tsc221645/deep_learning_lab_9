# Fase 4: baselines y modelos globales

## Modelos y protocolo

- **Naive, SeasonalNaive 24/168 y AutoETS(A,A,A):** se eligió AutoETS en lugar de MSTL por su estructura explícita de nivel, tendencia y estacionalidad diaria, y porque StatsForecast permite avanzar el estado con observaciones conocidas en cada origen. Se fija la estructura aditiva `AAA` para no hacer una búsqueda automática costosa sobre los 370 clientes. Los baselines estacionales usan el historial disponible antes del origen.
- **LightGBM:** estrategia directa, con un regresor por paso de `h=1..24`. Cada regresor recibe los rezagos 1 a 24, el rezago 168 y el calendario del timestamp que predice. Los orígenes de entrenamiento usan ventanas del split de entrenamiento; el conjunto de validación solo se usa para early stopping. El límite predeterminado de 256 ventanas por cliente se distribuye a lo largo del historial y puede cambiarse con `max_train_windows_per_client`.
- **MLP:** una red compartida para todos los clientes, con entrada histórica y calendario pasado/futuro, y salida conjunta de 24 horas.
- **DLinear:** separa tendencia y componente estacional con una media móvil de ventana 25 calculada dentro del historial observado. Cada componente tiene su propia proyección lineal.
- **NLinear:** resta el último valor de cada ventana, proyecta los cambios y restituye ese nivel para los 24 pasos.

Los modelos PyTorch usan `Trainer`, MSE en escala normalizada, early stopping por pérdida de validación y registros de MAE/MASE en kWh por época. Las opciones de escalado son `none`, `global` y `per_series`; sus estadísticas se ajustan solo con el panel de entrenamiento. LightGBM registra MAE, MSE y MASE por iteración. Todos los pronósticos se invierten a kWh y pasan por el mismo backtest y las mismas métricas de fase 3.

En esta implementación, StatsForecast y LightGBM ejecutan los baselines en CPU; solo los modelos PyTorch tienen una ruta CUDA. Si el requisito de GPU incluye también los baselines, hace falta cambiar su implementación o aclarar una excepción antes de afirmar cumplimiento completo.

El evaluador registra los parámetros entrenables de PyTorch. Para LightGBM, `params_total` y `params_trainable` cuentan las hojas de los árboles, que son los valores efectivos ajustados por los regresores.

El backtest mantiene los 92 orígenes diarios de prueba. LightGBM requiere 168 observaciones para construir el rezago semanal, por lo que la variante `L=48` solo se ejecuta en redes PyTorch. Las redes se comparan con `L=48, 168, 336` y los tres modos de normalización. La sensibilidad de historia de 3, 12 y 30 meses se ejecuta para SeasonalNaive 24/168, LightGBM y DLinear.

## Ejecución

Desde la raíz del repositorio:

```powershell
python -m src.experiments.phase4
```

La ejecución completa carga `data/processed/hourly.parquet` y `client_ids.csv`, entrena todos los modelos y variantes y guarda:

- pronósticos y métricas uniformes en `outputs/predictions/`;
- resumen de experimentos en `outputs/metrics/phase4_summary.csv` y `.json`;
- curvas de entrenamiento y pesos en `outputs/models/phase4/`.

Para validar la instalación y las interfaces con un origen y hasta ocho clientes, usa `--smoke`. No uses esas métricas como resultados de evaluación. Puedes limitar el trabajo, por ejemplo, con `--models dlinear lightgbm`.

Para Kaggle, sube el repositorio y los datos procesados, instala `requirements.txt` y ejecuta el mismo módulo desde el directorio del proyecto. El runner no cambia los IDs, splits ni las métricas.

## Archivos de implementación

- `src/models/statsforecast_adapters.py`
- `src/models/lightgbm_adapter.py`
- `src/models/torch_forecasters.py`
- `src/training/trainer.py`
- `src/experiments/phase4.py`
