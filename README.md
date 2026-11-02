# Forecasting de consumo eléctrico

Implementación del laboratorio de series de tiempo de CC3092. El enunciado de la guía lo identifica como Laboratorio 8, aunque el nombre del archivo original menciona Laboratorio 9.

## Requisitos

- Python 3.12 o 3.13 (las pruebas locales reportadas pasaron con 3.12)
- Git
- Aproximadamente 1 GB libre para el archivo comprimido, el archivo original y los datos procesados

Las versiones directas están fijadas en `requirements.txt`. En Windows se puede crear el entorno así:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Kaggle, sigue [INSTRUCTIVO_KAGGLE3.md](INSTRUCTIVO_KAGGLE3.md) para instalar el par compatible de PyTorch y `requirements-kaggle.txt`. El procesamiento lee el archivo en bloques, así que no requiere cargar los 678 MB completos en memoria.

## Datos

El proyecto usa [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), de Artur Trindade, UCI Machine Learning Repository, DOI [10.24432/C58C86](https://doi.org/10.24432/C58C86). UCI publica el ZIP de 249.2 MB y el archivo de texto de 678.1 MB. La licencia es CC BY 4.0.

Descarga y procesa los 370 clientes:

```powershell
python -m src.data.download --output-dir data/raw
python -m src.data.preprocess --input data/raw/LD2011_2014.txt --output data/processed/hourly.parquet --audit-dir data/processed --seed 42
```

Si el entorno no permite procesar los 370, fija un subconjunto reproducible con `--max-clients 100`. El archivo `data/processed/client_ids.csv` congela los IDs y se debe reutilizar en todos los modelos. Los datos brutos y el panel procesado no se versionan; los IDs y resúmenes pequeños de auditoría sí.

## Decisiones de datos

- El archivo tiene valores de potencia en kW cada 15 minutos. La energía horaria se calcula como `0.25 * suma(kW de los cuatro intervalos)` y queda en kWh.
- Las cuatro observaciones con marcas `HH:00`, `HH:15`, `HH:30` y `HH:45` forman la hora `HH:00`. Las marcas del archivo se agrupan por hora local, sin desplazar el índice. Las horas incompletas en los extremos del archivo se excluyen, no se imputan.
- Las marcas de tiempo representan la hora local portuguesa sin zona horaria. Se conserva la cuadrícula local de 24 horas, sin convertir a UTC ni inventar observaciones. UCI describe ceros en la hora omitida de marzo y una hora agregada en octubre. La auditoría confirma 24 horas por día entre 2012 y 2014, pero también encuentra lecturas no nulas en la hora de marzo para parte de los clientes; por eso se conservan los valores originales y se reporta la discrepancia en `dataset_audit.json`.
- Los ceros no se tratan como valores ausentes. UCI indica que algunos clientes todavía no existían al inicio y se representan con ceros, y que también hay ceros en la transición de marzo. Se reportan por cliente para inspección.
- La elegibilidad se determina por cobertura temporal y valores finitos desde el 1 de enero de 2012, no por consumo positivo. La auditoría valida 24 horas locales por día y ausencia de marcas duplicadas.

## Estructura

- `src/data/`: descarga, auditoría y transformación del dataset.
- `src/utils/`: utilidades de reproducibilidad.
- `tests/`: pruebas del tratamiento temporal y la conversión de unidades.
- `configs/`: parámetros iniciales compartidos.
- `data/raw/` y `data/processed/`: datos locales ignorados por Git.

Las fases se desarrollan en orden y cada fase se valida y confirma en su propio commit.

## Fase 2: EDA y ventanas

Ejecuta el análisis o abre `notebooks/laboratorio_series_tiempo.ipynb`:

```powershell
python -m src.data.eda
python -m pytest -q
```

El panel horario se carga por grupos Parquet en una matriz `clientes x horas`. La selección de covariables por defecto es cíclica (7 columnas); también está disponible one-hot (44 columnas). Se puede usar la serie original, z-score global o z-score por cliente; los estadísticos se ajustan solo entre 2012-01-01 y 2014-07-01. La variante por ventana usa únicamente el historial y conserva sus parámetros para invertir la escala antes de calcular métricas.

Las ventanas usan L=168, H=24 y stride 1. Se separan por fecha objetivo y las covariables futuras de calendario se entregan aparte del historial y del target. Los conteos de entrenamiento/validación/prueba y los 92 orígenes diarios de prueba quedan en `outputs/metrics/window_counts.csv`; los feriados portugueses se listan en `outputs/metrics/portugal_holidays.csv` y las figuras descriptivas quedan en `outputs/figures/eda/`.

## Fase 3: evaluación común

La interfaz de adaptadores, el backtest principal de 92 orígenes, las tres reevaluaciones con ajuste mensual, las métricas y las notas de APIs están documentadas en [docs/fase3_evaluacion.md](docs/fase3_evaluacion.md). El motor guarda predicciones Parquet y métricas por cliente, horizonte, hora y semana navideña en `outputs/predictions/`. Ejecuta las pruebas con:

```powershell
python -m pytest -q
```

Para el ajuste final de prueba, selecciona `best_epoch` con validación y configura `refit_train_validation=True`. La evaluación mensual crea un modelo nuevo al inicio de cada mes y excluye todos los datos de ese mes del ajuste inicial.

## Fase 4: modelos iniciales

La fase 4 incluye baselines StatsForecast, AutoETS, LightGBM directo a 24 pasos, MLP, DLinear y NLinear. La matriz de experimentos, las opciones de sensibilidad y los artefactos quedan descritos en [docs/fase4_modelos.md](docs/fase4_modelos.md). Para ejecutar:

```powershell
python -m src.experiments.phase4
```

Usa `--smoke` para una comprobación rápida, no para reportar métricas finales. La entrega de interfaces y decisiones para la siguiente fase está en [HANDOFF_A_B.md](HANDOFF_A_B.md).

## Trabajo del integrante B

Para ejecutarlo por etapas y conservar resultados entre sesiones de Kaggle,
sigue [INSTRUCTIVO_KAGGLE3.md](INSTRUCTIVO_KAGGLE3.md).

La fase 5 incluye tres estrategias LSTM. La fase 6 incluye NHITS, TCN,
PatchTST, TFT, DeepAR y Chronos-Bolt. Las fases 7 y 8 tienen runners para
sensibilidad, reajustes y comparaciones. El detalle está en
[EJECUCION_INTEGRANTE_B.md](EJECUCION_INTEGRANTE_B.md).

Las LSTM usan validación para elegir la configuración y las épocas; el
backtest de prueba no selecciona hiperparámetros. Al entrenar en GPU, el
adaptador PyTorch requiere CUDA y falla si no está disponible.

Los smoke de fase 4 y de los modelos de B produjeron 192 predicciones por
modelo, usando ocho clientes y un origen. Los metadatos registran CUDA en 12
corridas y CPU en 5. Naive, SeasonalNaive, AutoETS y LightGBM usan
implementaciones CPU en esta configuración y no tienen una ruta CUDA; por eso
no se debe afirmar que todos los modelos corrieron en GPU. Esas corridas solo
comprueban integración; no sirven para comparar modelos. La suite local actual
tiene 46 pruebas aprobadas.
Los resultados completos de prueba, las sensibilidades de fase 7 y el PDF final
no se han generado.

El informe de estado no sustituye al final. Para generarlo después de ejecutar
las pruebas, usa `python -m src.experiments.report_status --tests-passed N`;
el PDF y el Markdown quedan en `report/`.

El panel `data/processed/hourly.parquet` y los artefactos grandes de `outputs/`
se mantienen fuera de Git. Para generar las métricas finales, primero hay que
ejecutar todas las etapas desde el panel congelado y conservar esos artefactos.
Los documentos metodológicos de las fases 3 y 4 están en `docs/`.
En Kaggle, sube el código actualizado y los datos procesados, activa GPU e
Internet para Chronos y conserva `outputs/` entre sesiones.

### Corrida completa

El bloque siguiente enumera los módulos originales del proyecto y puede
exceder una sesión de Kaggle; para una ejecución práctica utiliza la guía
anterior, que separa configuraciones y explica los resultados faltantes de A.

Desde la raíz del proyecto, con GPU CUDA y dependencias instaladas:

```powershell
python -m src.data.download --output-dir data/raw
python -m src.data.preprocess --input data/raw/LD2011_2014.txt --output data/processed/hourly.parquet --audit-dir data/processed --seed 42
python -m pytest -q
python -m src.experiments.phase4
python -m src.experiments.phase5 select
python -m src.experiments.phase5 test
python -m src.experiments.phase6
python -m src.experiments.chronos_bolt
python -m src.experiments.phase7 history
python -m src.experiments.phase7 monthly
python -m src.experiments.phase7 local
python -m src.experiments.phase7 seeds
python -m src.experiments.phase8
```

La ejecución completa de estas etapas todavía está pendiente en este checkout.
La fase 8 requiere sus resultados y genera un borrador; no debe tomarse como
entrega final hasta validar las conclusiones y producir el PDF de cinco páginas
como máximo.
