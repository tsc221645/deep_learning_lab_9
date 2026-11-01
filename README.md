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

En Kaggle, instala las dependencias con `!pip install -r requirements.txt` y ejecuta los mismos módulos desde la raíz del proyecto. El procesamiento lee el archivo en bloques, así que no requiere cargar los 678 MB completos en memoria.

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

## Trabajo del integrante B (preparado, sin ejecutar)

Para ejecutar todas las fases en Kaggle, seguir [KAGGLE_EJECUCION.md](KAGGLE_EJECUCION.md). Incluye instalación separada de PyTorch, transferencia del panel, comandos por etapa y recuperación de `outputs/` entre sesiones.

El estado observado, los fallos de VRAM y las instrucciones para otra persona
están en [EJECUCION_INTEGRANTE_B.md](EJECUCION_INTEGRANTE_B.md).

El panel `data/processed/hourly.parquet` existe localmente, pero está ignorado
por Git y debe transferirse aparte. El repositorio no incluye los pronósticos y
métricas completos de la fase 4. Tampoco están presentes los documentos `docs/fase3_evaluacion.md`
y `docs/fase4_modelos.md` citados en el traspaso. Por tanto, las fases 1–4 tienen
código y pruebas, pero su evaluación completa no está acreditada en este checkout.

Se añadieron las tres estrategias LSTM de la fase 5 (recursiva, multi-salida y
seq2seq) y el adaptador zero-shot de Chronos-Bolt. El entrenamiento recursivo y
seq2seq puede usar teacher forcing; validación y prueba realimentan únicamente
pronósticos. La búsqueda de L usa la validación para elegir una configuración por
estrategia y guarda esa elección antes de evaluar prueba. El backtest final
reajusta con entrenamiento + validación usando el número de epochs elegido.

### Primera comprobación recomendada

No iniciar aún todas las corridas. Primero confirmar Python 3.12/3.13 y una GPU CUDA,
instalar dependencias, preparar el panel si no fue transferido y correr las pruebas
y una integración pequeña. El primer ejecutor ya pasó 42 pruebas con Python 3.12.

```powershell
py -0p
nvidia-smi
python -m venv .venv
$labPy = '.\.venv\Scripts\python.exe'
& $labPy -m pip install -r requirements.txt
& $labPy -m src.data.download --output-dir data/raw
& $labPy -m src.data.preprocess --input data/raw/LD2011_2014.txt --output data/processed/hourly.parquet --audit-dir data/processed --seed 42
& $labPy -m pytest -q
& $labPy -m src.experiments.phase4 --smoke
& $labPy -m src.experiments.smoke_b --models lstm_recursive
```

`smoke_b` usa ocho clientes, un origen y entrenamiento mínimo. Para comprobar
después un adaptador concreto, sustituye `lstm_recursive` por `nhits`, `tcn`,
`patchtst`, `tft` o `deepar`. `chronos` descarga pesos públicos la primera vez;
pruébalo preferentemente en Kaggle con Internet habilitado. Ninguna salida
`--smoke` sirve como resultado final. Detente si falla cualquiera de los pasos
y conserva el error completo; no continúes con la fase siguiente.

Si ejecutas en Kaggle, copia o sube **este estado actualizado del repositorio**:
los cambios locales sin commit no aparecen al clonar GitHub. Activa GPU e
Internet y verifica `python --version` y CUDA antes de instalar las versiones
fijadas. Guarda `data/processed/` y `outputs/` entre sesiones.

Para reproducirlo en un entorno remoto con GPU CUDA, desde la raíz del proyecto
y tras instalar `requirements.txt`, sigue este orden:

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

Estos comandos de B no se han ejecutado ni verificado en este checkout. La fase
8 exige resultados completos y se detiene si faltan modelos, clientes, fechas,
cuantiles o los experimentos de historia. Genera CSV, figuras y un borrador en
`report/informe_borrador.md`; ese borrador necesita interpretación humana, el
enlace Git real y conversión/revisión a un PDF de cinco páginas como máximo.
El notebook incluye las secciones de B con `RUN_EXPENSIVE = False` por defecto.
No se deben presentar conclusiones finales hasta ejecutar y revisar los
artefactos exigidos.
