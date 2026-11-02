# Ejecución completa del laboratorio en Kaggle

Esta guía usa el código del repositorio actual y el panel horario ya procesado. **No se han ejecutado las corridas completas en Kaggle**. Los smoke hechos en la computadora original cubrieron LSTM, NHITS, TCN y PatchTST; TFT y DeepAR fallaron por VRAM antes de limitar los lotes, y Chronos sigue sin comprobarse.

## Punto actual de la ejecución en Kaggle

El usuario ya instaló las dependencias y ejecutó `python -m pytest -q` en Kaggle: **43 pruebas pasadas en 19.57 s**. Los 15 avisos proceden de `anyio`/pytest y `matplotlib`/pyparsing; son advertencias de deprecación, no fallos de pruebas. El problema inicial para localizar el panel en `/kaggle/input` ya quedó resuelto: las corridas de fase 4 produjeron predicciones completas. No repetir instalación, copia del panel ni pruebas en esta sesión.

**Actualización de avance:** el usuario ya compartió resultados completos de `lightgbm-h30`, `lightgbm-h3`, `lightgbm-h12`, las tres configuraciones principales MLP/DLinear/NLinear y `dlinear-L168-per_series-h3`, cada una con 816 960 predicciones de 370 clientes. La tercera línea de fase 4 seguía activa al reportarlo. Las advertencias `FutureWarning` de pandas no detienen el entrenamiento.

**Por qué se ejecutó fase 4 otra vez:** `HANDOFF_A_B.md` afirma que la fase 4 estaba completa, pero el checkout entregado no contenía sus pronósticos y métricas completos. Los archivos `.txt` bajo `outputs/models/phase4/lightgbm-direct/lightgbm-h30-smoke/` son modelos de la prueba pequeña: su `phase4_smoke_summary.csv` registra solo **192 predicciones, 8 clientes y 1 origen** por modelo. Los resultados actuales de Kaggle usan **816 960 predicciones, 370 clientes y 92 orígenes**; son las corridas completas necesarias para la comparación final. No es posible afirmar que A nunca ejecutó la fase 4 en otra máquina: lo verificable es que los artefactos completos no estaban en este repositorio local y `outputs/` está mayormente ignorado por Git. La fase 4 corresponde originalmente a A; se reprodujo en Kaggle para disponer de sus referencias antes de ejecutar B (fases 5–8).

## 1. Preparar lo que se subirá

1. Publicar los cambios actuales del repositorio en GitHub o subir un ZIP con el código actualizado como Kaggle Dataset. Antes de clonar en Kaggle, comprobar que los archivos `requirements-kaggle.txt`, `src/experiments/phase7.py` y esta guía estén en GitHub. Un commit local no basta: debe haber `git push`. Si se usa ZIP, extraerlo en `/kaggle/working` y trabajar allí.
2. En Kaggle, abrir **Datasets → New Dataset**, subir `data/processed/hourly.parquet`, `data/processed/client_ids.csv`, `data/processed/dataset_audit.json` y, si se desea, `per_client_audit.csv`; poner un nombre reconocible, elegir **Private** y crear el Dataset. El panel debe corresponder a los IDs congelados. No subir el archivo bruto de 711 MB si ya se dispone del Parquet.
3. Abrir **Code → New Notebook** en Kaggle. En **Settings**, seleccionar GPU (preferiblemente T4 x2; el código usa una GPU) e **Internet On**. En el panel **Input → Add Input**, buscar el Dataset privado anterior y añadirlo. Los nombres exactos de estas opciones pueden cambiar en la interfaz.
4. En Kaggle, `/kaggle/input` es la entrada del Dataset y se usa como solo lectura; el código y los resultados se escriben en `/kaggle/working`. Los experimentos completos pueden necesitar varias sesiones. Kaggle documenta sesiones GPU de hasta 12 horas y hasta 20 GB persistidos en `/kaggle/working`; la disponibilidad y cuota efectiva dependen de la cuenta.

## 2. Celdas de preparación del Notebook de Kaggle

Ejecutar estas celdas **en orden**. Sustituir `URL_DEL_REPOSITORIO` y `NOMBRE_DEL_DATASET` por los valores reales. Si se usa el repositorio configurado en este checkout, la URL es `https://github.com/tsc221645/deep_learning_lab_9.git`. Confirmar antes que los cambios recientes estén publicados.

```python
!git clone https://github.com/tsc221645/deep_learning_lab_9.git /kaggle/working/deep_learning_lab_9
%cd /kaggle/working/deep_learning_lab_9
!git rev-parse HEAD
!python --version
!nvidia-smi
```

```python
# Instalar el trío binario compatible. No mezclar torch 2.11 con torchvision 0.28.
%pip install -q torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
%pip install -q -r requirements-kaggle.txt
```

Si pip muestra avisos sobre paquetes preinstalados de Kaggle, comprobar primero si la instalación terminó con `Successfully installed` y si la siguiente celda de importaciones funciona. Esta lista se ajustó a `numpy==2.2.6` y `matplotlib==3.10.0` para respetar `numba 0.61.2` y `ydata-profiling 4.18.4` del entorno reportado. `bigframes`, `dopamine-rl`, `gradio` y `moviepy` no forman parte de este laboratorio; sus avisos no son una razón para reinstalar todo el entorno. Si aparece un `ERROR` final o falla una importación del proyecto, conservar el error completo.

**Reiniciar el kernel** después de instalar, desde el menú del Notebook; luego volver a `%cd /kaggle/working/deep_learning_lab_9`. Esto evita que queden módulos anteriores cargados en memoria. Comprobar las versiones y CUDA:

```python
%cd /kaggle/working/deep_learning_lab_9
import torch, torchvision
print(torch.__version__, torchvision.__version__, torch.cuda.is_available())
assert torch.cuda.is_available()
assert torch.__version__.split('+')[0] == '2.11.0'
assert torchvision.__version__.split('+')[0] == '0.26.0'
from torchvision.ops import nms  # Comprueba el operador que falló antes.
import numpy, pandas, matplotlib, numba
import statsforecast, neuralforecast, lightgbm, chronos
print(numpy.__version__, matplotlib.__version__, numba.__version__)
```

Los nombres visibles del Dataset no siempre coinciden con su carpeta dentro de `/kaggle/input` (por ejemplo, Kaggle puede cambiar mayúsculas por minúsculas). Buscar el archivo real y copiarlo junto con los IDs **desde la misma carpeta**. Ejecutar esta celda desde la raíz del repositorio; no hace falta volver a correr `pytest` si ya mostró 43 pruebas pasadas. **No usar un panel generado con otra selección de clientes.**

```python
%cd /kaggle/working/deep_learning_lab_9
from pathlib import Path
from shutil import copy2

matches = sorted(Path('/kaggle/input').rglob('hourly.parquet'))
print('Paneles encontrados:', matches)
if len(matches) != 1:
    print('Datasets conectados:', list(Path('/kaggle/input').iterdir()))
    raise FileNotFoundError('Debe haber exactamente un hourly.parquet en Input; revisar Add Input o el contenido del Dataset')

source = matches[0].parent
target = Path('data/processed')
target.mkdir(parents=True, exist_ok=True)
for filename in ('hourly.parquet', 'client_ids.csv'):
    file = source / filename
    if not file.is_file():
        raise FileNotFoundError(f'Falta {file}; subirlo al mismo Dataset que el panel')
    copy2(file, target / filename)
audit = source / 'dataset_audit.json'
if audit.is_file():
    copy2(audit, target / audit.name)
print('Copiado desde:', source)
print('Archivos de trabajo:', sorted(p.name for p in target.iterdir()))
```

Si `Paneles encontrados: []`, abrir **Input → Add Input** y comprobar que el Dataset esté conectado y contenga `hourly.parquet`; no basta con haberlo creado. Para inspeccionar manualmente nombres y carpetas, ejecutar `!ls -lah /kaggle/input` y `!find /kaggle/input -maxdepth 4 -type f | head -40`. Si el panel aparece en otra subcarpeta, la celda anterior lo encontrará igual. La línea `cp` del ejemplo anterior para `dataset_audit.json` también necesitaba un segundo argumento de destino; ya está corregida aquí.

Si no se tiene el panel, se puede habilitar Internet y ejecutar `python -m src.data.download --output-dir data/raw` seguido de `python -m src.data.preprocess --input data/raw/LD2011_2014.txt --output data/processed/hourly.parquet --audit-dir data/processed --seed 42`. Guardar después el Parquet como Dataset privado para no repetir la descarga.

## 3. Primera comprobación de GPU y adaptadores

**Siguiente paso desde el punto actual:** comprobar que la sesión sigue con GPU y que los datos procesados están en la ruta esperada. Ejecutar esta celda:

```python
%cd /kaggle/working/deep_learning_lab_9
import torch
print("GPU disponible:", torch.cuda.is_available())
!nvidia-smi
!ls -lh data/processed/hourly.parquet data/processed/client_ids.csv
```

Si `GPU disponible` es `False` o falta uno de los dos archivos, corregir Settings/Input y la copia del panel antes de continuar. Si es `True` y ambos archivos existen, ejecutar **solo la primera línea** del bloque siguiente (`phase4 --smoke`) y comprobar su resultado. Luego avanzar una línea por vez; cada línea puede ir en una celda distinta. Conservar el traceback completo si alguna falla. Los smoke tienen ocho clientes y un origen; **sus métricas no son resultados finales**.

```python
!python -m src.experiments.phase4 --smoke
!python -m src.experiments.smoke_b --models lstm_recursive lstm_multioutput lstm_seq2seq
!python -m src.experiments.smoke_b --models nhits tcn patchtst
!python -m src.experiments.smoke_b --models tft
!python -m src.experiments.smoke_b --models deepar
!python -m src.experiments.smoke_b --models chronos
```

Si TFT o DeepAR siguen agotando VRAM, anotar el modelo, comando, GPU, `nvidia-smi` y traceback. Los lotes de NeuralForecast ahora están limitados (`valid_batch_size=8`, `windows_batch_size=64`, `inference_windows_batch_size=32`; smoke: 2/8/8), pero **no se ha medido aún** su suficiencia en Kaggle. No reducir clientes solo para un modelo: todos los resultados finales deben compartir el mismo panel, con 370 clientes si es viable o mínimo 100 para todos.

## 4. Corridas finales, en orden

Cada línea es una celda `!python ...` independiente. Así se ve qué etapa terminó y se pueden guardar versiones. **No usar `Run All` del notebook principal** como sustituto: sus celdas costosas están desactivadas por defecto. Tampoco lanzar todas las líneas a la vez; varias pueden tardar horas.

### Fase 4: resultados de A que faltan en este checkout

La comparación final exige las predicciones completas de los ocho modelos de A. Si no se reciben de A, correr la fase 4 en Kaggle. Puede dividirse por familia; los artefactos quedan separados por `config_id`:

**Uso de CPU/GPU:** `naive`, `seasonal_naive_24`, `seasonal_naive_168`, `auto_ets` y `lightgbm` se ejecutan en CPU. `mlp`, `dlinear` y `nlinear` exigen CUDA y entrenan en GPU, aunque la preparación de datos y parte de la evaluación también ocupen CPU. Si se pegan las tres líneas en una sola celda, Jupyter las ejecuta **en orden, una después de la otra**; eso es válido. Hasta que termine la segunda línea, no se habrá iniciado ningún modelo de GPU. Para identificar con claridad el avance y guardar resultados entre etapas, se recomienda una celda por línea. No repetir una corrida que siga activa.

**Tiempo y límite de sesión:** las tres líneas originales sin filtro abarcan 40 configuraciones completas: 8 baselines/estadísticos, 3 LightGBM y 29 variantes de MLP/DLinear/NLinear. Para obtener primero los ocho modelos principales, la opción nueva `--primary-only` reduce esas líneas a 4, 1 y 3 configuraciones respectivamente. Las variantes adicionales de fase 4 deben ejecutarse después si se necesitan para la sensibilidad de A; `--primary-only` no las acredita. AutoETS sigue siendo costoso aun con el filtro, porque trabaja por cliente y por los 92 orígenes. No hay una estimación fiable sin saber qué configuración está en curso. Kaggle documenta un máximo de 12 horas por sesión CPU/GPU, por lo que **no se debe asumir que las tres líneas terminarán juntas en una sesión**. Las corridas completadas guardan artefactos separados en `outputs/predictions/`; conservarlos antes de cambiar de sesión. Esta opción requiere actualizar el repositorio en Kaggle antes de usarla; no afecta una celda que ya está corriendo.

```python
!python -m src.experiments.phase4 --primary-only --models naive seasonal_naive_24 seasonal_naive_168 auto_ets
!python -m src.experiments.phase4 --primary-only --models lightgbm
!python -m src.experiments.phase4 --primary-only --models mlp dlinear nlinear
```

`outputs/metrics/phase4_summary.csv` se acumula por `config_id` entre invocaciones, y la fase 8 lee los artefactos de `outputs/predictions/`. Conservar `outputs/` completo.

Si el indicador **Output** de Kaggle no cambia mientras el comando trabaja, no concluir que se perdieron los resultados. El evaluador escribe `predictions.parquet`, `experiment.json` y las tablas CSV **al terminar cada configuración**; el resumen de fase 4 se actualiza al terminar el comando de cada familia. Cuando la celda actual termine, comprobar el espacio y los archivos con:

```python
!du -sh outputs
!find outputs/predictions -name experiment.json | wc -l
!ls -lh outputs/metrics/phase4_summary.csv outputs/predictions/mlp/mlp-L168-per_series-h30/predictions.parquet
```

No ejecutar esas celdas si Kaggle todavía tiene ocupada la celda del entrenamiento. La interfaz puede actualizar su indicador de almacenamiento con retraso; los archivos reales en `/kaggle/working/deep_learning_lab_9/outputs` son la comprobación útil.

### Fases 5 y 6

```python
!python -m src.experiments.phase5 select
!python -m src.experiments.phase5 test
!python -m src.experiments.phase6 --models NHITS
!python -m src.experiments.phase6 --models TCN
!python -m src.experiments.phase6 --models PatchTST
!python -m src.experiments.phase6 --models TFT
!python -m src.experiments.phase6 --models DeepAR
!python -m src.experiments.chronos_bolt
```

La fase 6 acumula `outputs/metrics/phase6_summary.csv` por modelo. Chronos descarga los pesos la primera vez y requiere Internet o caché. Las predicciones finales de cada modelo deben tener **370 × 92 × 24 = 816 960 filas** si se usa el panel completo. Para 100 clientes serían 220 800. La validación para seleccionar modelos no debe mirar la prueba.

### Fase 7: experimentos adicionales

La sensibilidad a historia se puede dividir en cinco grupos, uno por familia. Cada grupo contiene 3, 12 y 30 meses; `phase7_history.csv` se guarda después de cada configuración y se reutilizan las filas ya terminadas al reiniciar el comando. El grupo `recurrent` usa la estrategia LSTM elegida en validación.

```python
!python -m src.experiments.phase7 history --history-models seasonal_naive_24
!python -m src.experiments.phase7 history --history-models lightgbm
!python -m src.experiments.phase7 history --history-models dlinear
!python -m src.experiments.phase7 history --history-models recurrent
!python -m src.experiments.phase7 history --history-models patchtst
!python -m src.experiments.phase7 monthly
!python -m src.experiments.phase7 local
!python -m src.experiments.phase7 seeds
```

`monthly` compara ajuste fijo y reajuste al inicio de octubre, noviembre y diciembre; `local` ajusta diez modelos locales; `seeds` usa 42, 43 y 44. Estos tres comandos aún son unidades grandes: si una sesión termina a mitad, el comando incompleto debe repetirse. Los outputs completos de comandos anteriores se pueden reutilizar.

La selección del mejor modelo deep de fase 7 guarda cada candidato de validación en `outputs/metrics/phase7_deep_candidates.json` y la selección del mejor modelo propio en `phase7_own_candidates.json`. Restaurar también esos archivos entre sesiones para continuar una selección interrumpida sin repetir candidatos ya terminados.

### Fase 8 y entrega

```python
!python -m src.experiments.phase8 --repo-url https://github.com/tsc221645/deep_learning_lab_9
!ls outputs/metrics/comparison_final.csv outputs/figures/final report/informe_borrador.md
```

La fase 8 comprueba los experimentos de los modelos principales, el número de orígenes y la igualdad de clientes, fechas y valores reales. Genera tabla, figuras y **borrador Markdown**, todavía no el PDF final. Leer los resultados reales, completar las respuestas exigidas en la guía, añadir el enlace Git real, convertir a PDF y comprobar que tenga **máximo cinco páginas**. No inventar resultados faltantes. `MASE_H168` debe figurar N/A si no se evaluó ese horizonte.

## 5. Guardar avances y continuar en otra sesión

1. Tras un bloque terminado, guardar una versión del Notebook **con archivos de salida** o crear un Dataset privado de los resultados en `/kaggle/working/deep_learning_lab_9/outputs`. Verificar en la página de la versión que los archivos aparezcan. **Save & Run All inicia una sesión limpia y vuelve a ejecutar las celdas**, por lo que conviene usar notebooks pequeños, uno por bloque, o crear un Dataset de salidas tras cada bloque.
2. En el siguiente Notebook, añadir como **Input** el Dataset/Notebook con salidas anteriores. Clonar exactamente el mismo commit del código, copiar el panel procesado y **restaurar `outputs/` antes de correr la siguiente fase**. Ejemplo, sustituyendo la ruta real indicada por Kaggle:

```python
!mkdir -p /kaggle/working/deep_learning_lab_9/outputs
!cp -a /kaggle/input/NOMBRE_DEL_DATASET_DE_RESULTADOS/outputs/. /kaggle/working/deep_learning_lab_9/outputs/
!test -f /kaggle/working/deep_learning_lab_9/outputs/metrics/phase5_selected.json
```

3. Si el input de resultados no tiene carpeta `outputs` sino directamente `metrics`, `predictions`, etc., copiar esas carpetas al `outputs/` de trabajo. **Nunca escribir dentro de `/kaggle/input`**.
4. Para devolverme los resultados, descargar o compartir como Dataset/Notebook output `outputs/metrics/`, `outputs/predictions/`, `outputs/models/` (logs), `outputs/figures/`, `report/`, `data/processed/client_ids.csv`, `dataset_audit.json` y el log de comandos/errores. Guardar también `python --version`, versiones de torch/torchvision, modelo GPU y `git rev-parse HEAD`.

## 6. Criterio de finalización

Hasta que las corridas reales terminen, el proyecto **está preparado para ejecutar, no validado experimentalmente ni listo para entregar**. En especial, TFT, DeepAR y Chronos necesitan nueva prueba en Kaggle; el informe final requiere los resultados completos y revisión humana. Si una etapa falla, enviar el traceback y los artefactos existentes antes de cambiar hiperparámetros o clientes.

Referencias de plataforma: [documentación oficial de Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) y [combinaciones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/).
