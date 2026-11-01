# Ejecución completa del laboratorio en Kaggle

Esta guía usa el código del repositorio actual y el panel horario ya procesado. **No se han ejecutado las corridas completas en Kaggle**. Los smoke hechos en la computadora original cubrieron LSTM, NHITS, TCN y PatchTST; TFT y DeepAR fallaron por VRAM antes de limitar los lotes, y Chronos sigue sin comprobarse.

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

**Reiniciar el kernel** después de instalar, desde el menú del Notebook; luego volver a `%cd /kaggle/working/deep_learning_lab_9`. Esto evita que queden módulos anteriores cargados en memoria. Comprobar las versiones y CUDA:

```python
%cd /kaggle/working/deep_learning_lab_9
import torch, torchvision
print(torch.__version__, torchvision.__version__, torch.cuda.is_available())
assert torch.cuda.is_available()
assert torch.__version__.split('+')[0] == '2.11.0'
assert torchvision.__version__.split('+')[0] == '0.26.0'
from torchvision.ops import nms  # Comprueba el operador que falló antes.
```

Copiar los datos desde la ruta que Kaggle muestre en **Input**. Si no coincide con el ejemplo, listar `/kaggle/input` y reemplazar la ruta. **No usar un panel generado con otra selección de clientes.**

```python
!find /kaggle/input -maxdepth 4 -name hourly.parquet -o -name client_ids.csv
!mkdir -p data/processed
!cp /kaggle/input/NOMBRE_DEL_DATASET/hourly.parquet data/processed/hourly.parquet
!cp /kaggle/input/NOMBRE_DEL_DATASET/client_ids.csv data/processed/client_ids.csv
!cp /kaggle/input/NOMBRE_DEL_DATASET/dataset_audit.json data/processed/dataset_audit.json
!python -m pytest -q
```

Si no se tiene el panel, se puede habilitar Internet y ejecutar `python -m src.data.download --output-dir data/raw` seguido de `python -m src.data.preprocess --input data/raw/LD2011_2014.txt --output data/processed/hourly.parquet --audit-dir data/processed --seed 42`. Guardar después el Parquet como Dataset privado para no repetir la descarga.

## 3. Primera comprobación de GPU y adaptadores

Ejecutar cada comando en una celda separada y conservar el traceback completo si falla. Los smoke tienen ocho clientes y un origen; **sus métricas no son resultados finales**.

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

```python
!python -m src.experiments.phase4 --models naive seasonal_naive_24 seasonal_naive_168 auto_ets
!python -m src.experiments.phase4 --models lightgbm
!python -m src.experiments.phase4 --models mlp dlinear nlinear
```

`outputs/metrics/phase4_summary.csv` se acumula por `config_id` entre invocaciones, y la fase 8 lee los artefactos de `outputs/predictions/`. Conservar `outputs/` completo.

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
