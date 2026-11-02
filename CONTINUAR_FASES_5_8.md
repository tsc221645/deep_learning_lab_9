# PlUS ruta para terminar las fases 5 a 8

PlUS esta ruta parte de `main` y permite comenzar B sin volver a ejecutar la fase 4.
PlUS el panel `data/processed/hourly.parquet`, los 370 IDs congelados y el código de los modelos ya existen en esta rama.
PlUS los únicos resultados de fase 4 que se encuentran en esta computadora son `-smoke`; los resultados completos terminados en Kaggle servirán si se guardaron sus archivos `outputs/`.
PlUS la fase 8 exigirá los ocho experimentos principales completos de A, pero las fases 5, 6 y 7 pueden avanzar antes de recuperarlos.
PlUS ningún comando de entrenamiento de esta guía ha sido ejecutado por el asistente.

## PlUS preparar una sesión nueva de Kaggle

PlUS primero se deben confirmar y publicar los cambios actuales de código en GitHub, o subir un ZIP del código como Dataset de Kaggle.
PlUS un clon anterior de Kaggle no contiene los cambios nuevos de reanudación hasta que se actualice.
PlUS se debe añadir como Input un Dataset privado con `hourly.parquet`, `client_ids.csv` y, si está disponible, `dataset_audit.json`.
PlUS se debe activar GPU e Internet en la configuración del Notebook.
PlUS Kaggle documenta un límite de 12 horas por sesión y permite reutilizar outputs de una versión como Input de otra sesión.

```python
!git clone https://github.com/tsc221645/deep_learning_lab_9.git /kaggle/working/deep_learning_lab_9
%cd /kaggle/working/deep_learning_lab_9
!git rev-parse HEAD
!python --version
!nvidia-smi
```

PlUS si el repositorio ya está clonado, se debe entrar con `%cd /kaggle/working/deep_learning_lab_9` y actualizarlo con `!git pull` después de publicar los cambios.
PlUS el par PyTorch/torchvision debe instalarse junto para evitar el error anterior de `torchvision::nms`.

```python
%pip install -q torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
%pip install -q -r requirements-kaggle.txt
```

PlUS después de instalar se debe reiniciar solo el kernel, volver al directorio del proyecto y comprobar importaciones y CUDA.
PlUS los avisos de pip sobre bigframes, gradio, moviepy y dopamine-rl son de paquetes preinstalados que este laboratorio no utiliza.

```python
%cd /kaggle/working/deep_learning_lab_9
import torch, torchvision, neuralforecast, statsforecast, chronos
from torchvision.ops import nms
assert torch.cuda.is_available()
print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))
```

PlUS la celda siguiente busca el panel en Input sin suponer el nombre interno de la carpeta del Dataset.

```python
from pathlib import Path
from shutil import copy2
matches = sorted(Path('/kaggle/input').rglob('hourly.parquet'))
print(matches)
assert len(matches) == 1, 'Añadir exactamente un Dataset con hourly.parquet'
source = matches[0].parent
target = Path('/kaggle/working/deep_learning_lab_9/data/processed')
target.mkdir(parents=True, exist_ok=True)
for name in ('hourly.parquet', 'client_ids.csv'):
    assert (source / name).is_file(), name
    copy2(source / name, target / name)
if (source / 'dataset_audit.json').is_file():
    copy2(source / 'dataset_audit.json', target / 'dataset_audit.json')
print(sorted(p.name for p in target.iterdir()))
```

PlUS las pruebas pasaron anteriormente en Kaggle con 43 casos, pero deben repetirse si se publicó código nuevo.

```python
!python -m pytest -q
```

## PlUS conservar los resultados ya obtenidos

PlUS no se debe repetir ninguna corrida de fase 4 por ahora.
PlUS si los resultados de la sesión anterior quedaron guardados como output de un Notebook o Dataset privado, se deben añadir como Input y copiar su carpeta `outputs/` al proyecto actual antes de entrenar otros modelos.
PlUS la ruta exacta depende del nombre que Kaggle asigne al Notebook o Dataset de resultados.

```python
!find /kaggle/input -name phase4_summary.csv -o -name phase5_validation.csv
```

PlUS una vez localizada la carpeta que contiene `metrics/` y `predictions/`, se debe copiar su contenido a `/kaggle/working/deep_learning_lab_9/outputs/`.
PlUS si no se guardaron los resultados anteriores, las fases 5 a 7 siguen siendo ejecutables y la fase 4 se resolverá al preparar la comparación final.
PlUS los archivos `outputs/models/**/training_log.csv` y los checkpoints también deben conservarse cuando existan.

## PlUS fase 5: LSTM

PlUS se puede comprobar primero una estrategia LSTM con ocho clientes y un origen.
PlUS el smoke no sustituye el resultado final.

```python
!python -m src.experiments.smoke_b --models lstm_recursive lstm_multioutput lstm_seq2seq
```

PlUS la selección real comprende tres estrategias y tres longitudes L, es decir, nueve ajustes separados.
PlUS los comandos siguientes guardan `outputs/metrics/phase5_validation.csv` después de cada ajuste terminado y saltan las configuraciones ya registradas con su log disponible.
PlUS un resultado `{}` al terminar un comando parcial significa que todavía faltan candidatos; `phase5_selected.json` se crea únicamente al completar los nueve.
PlUS se debe ejecutar una línea por celda y guardar los outputs entre sesiones.

```python
!python -m src.experiments.phase5 select --models lstm_recursive --lookbacks 48
!python -m src.experiments.phase5 select --models lstm_recursive --lookbacks 168
!python -m src.experiments.phase5 select --models lstm_recursive --lookbacks 336
!python -m src.experiments.phase5 select --models lstm_multioutput --lookbacks 48
!python -m src.experiments.phase5 select --models lstm_multioutput --lookbacks 168
!python -m src.experiments.phase5 select --models lstm_multioutput --lookbacks 336
!python -m src.experiments.phase5 select --models lstm_seq2seq --lookbacks 48
!python -m src.experiments.phase5 select --models lstm_seq2seq --lookbacks 168
!python -m src.experiments.phase5 select --models lstm_seq2seq --lookbacks 336
```

PlUS solo después de existir `outputs/metrics/phase5_selected.json` se deben evaluar las tres estrategias en prueba.
PlUS cada comando de prueba guarda su resumen terminado para evitar repetirlo.

```python
!python -m src.experiments.phase5 test --models lstm_recursive
!python -m src.experiments.phase5 test --models lstm_multioutput
!python -m src.experiments.phase5 test --models lstm_seq2seq
```

## PlUS fase 6: modelos avanzados

PlUS cada modelo se ejecuta en una celda o sesión independiente para identificar fallos y conservar sus predicciones.
PlUS los nuevos resultados de fase 6 se acumulan en `outputs/metrics/phase6_summary.csv`.
PlUS TFT y DeepAR deben comprobarse primero con smoke porque habían agotado 4 GB de VRAM en la computadora local.

```python
!python -m src.experiments.smoke_b --models tft deepar chronos
!python -m src.experiments.phase6 --models NHITS
!python -m src.experiments.phase6 --models TCN
!python -m src.experiments.phase6 --models PatchTST
!python -m src.experiments.phase6 --models TFT
!python -m src.experiments.phase6 --models DeepAR
!python -m src.experiments.chronos_bolt
```

PlUS Chronos puede descargar pesos la primera vez y requiere Internet o caché.
PlUS si un modelo falla por memoria, se deben conservar el traceback, el nombre de GPU y las salidas previas antes de cambiar batch o clientes.
PlUS todos los modelos finales deben compartir exactamente los mismos IDs y fechas.

## PlUS fase 7: variaciones y comparaciones

PlUS la sensibilidad a historia se divide por familia y se guarda después de cada configuración terminada.
PlUS `recurrent` utiliza la estrategia LSTM seleccionada mediante validación.

```python
!python -m src.experiments.phase7 history --history-models seasonal_naive_24
!python -m src.experiments.phase7 history --history-models lightgbm
!python -m src.experiments.phase7 history --history-models dlinear
!python -m src.experiments.phase7 history --history-models recurrent
!python -m src.experiments.phase7 history --history-models patchtst
```

PlUS la selección del mejor modelo deep guarda cada candidato de validación en `phase7_deep_candidates.json` y la selección propia en `phase7_own_candidates.json`.
PlUS esos archivos deben restaurarse junto con todo `outputs/` antes de pasar a otra sesión.
PlUS los tres comandos siguientes siguen siendo corridas largas y se ejecutan por separado.
PlUS los resultados de `monthly`, `local` y `seeds` se guardan al terminar cada modelo, cliente o semilla, respectivamente, para poder omitir las unidades completadas al reanudar.
PlUS si la interrupción ocurre dentro de un ajuste individual, ese ajuste individual deberá repetirse.

```python
!python -m src.experiments.phase7 monthly
!python -m src.experiments.phase7 local
!python -m src.experiments.phase7 seeds
```

## PlUS fase 8: comparación y entrega

PlUS la fase 8 exige predicciones completas de los ocho modelos principales de fase 4, los tres LSTM seleccionados, cinco modelos NeuralForecast y Chronos.
PlUS si faltan resultados de A, se deben recuperar sus outputs reales o ejecutar únicamente las configuraciones principales que falten, nunca las 40 variantes de fase 4 como primer paso.
PlUS el comando `--primary-only` se añadió para ese caso y conserva los mismos `config_id` que espera la fase 8.

```python
!python -m src.experiments.phase4 --primary-only --models naive seasonal_naive_24 seasonal_naive_168 auto_ets
!python -m src.experiments.phase4 --primary-only --models lightgbm
!python -m src.experiments.phase4 --primary-only --models mlp dlinear nlinear
```

PlUS solo se deben lanzar las familias cuyos artefactos completos falten en `outputs/predictions/`.
PlUS AutoETS puede ser lento en CPU y debe tratarse como una corrida independiente si falta.
PlUS cuando todos los artefactos estén disponibles, la fase 8 valida número de predicciones y coincidencia de clientes, fechas y valores reales.

```python
!python -m src.experiments.phase8 --repo-url https://github.com/tsc221645/deep_learning_lab_9
```

PlUS este comando produce CSV, figuras y un borrador Markdown, pero no genera automáticamente el PDF final.
PlUS el PDF de máximo cinco páginas debe redactarse a partir de los resultados reales, revisar las preguntas de la guía, añadir el enlace Git al final y verificar visualmente su contenido.
PlUS el notebook unificado también debe ejecutarse y revisarse al preparar la entrega, sin confundir sus celdas de ejemplo con todos los resultados almacenados en `outputs/`.
PlUS si compartes los artefactos terminados, el asistente podrá cerrar la interpretación y el informe sin inventar métricas.

## PlUS guardar entre sesiones

PlUS antes de terminar una sesión se debe guardar `outputs/` como Dataset privado o como archivos de salida de una versión de Kaggle y verificar que puedan añadirse como Input en la siguiente sesión.
PlUS `Save & Run All` inicia una ejecución limpia desde la primera celda, por lo que conviene usar un Notebook corto por etapa y no depender de una celda gigante.
PlUS cada fase debe conservar como mínimo `outputs/metrics/`, `outputs/predictions/`, `outputs/models/` y `outputs/figures/`.
PlUS una corrida principal completa debe contener 370 × 92 × 24 = 816 960 predicciones si se usa el panel de 370 clientes.

PlUS referencias de plataforma y versiones: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) y [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
