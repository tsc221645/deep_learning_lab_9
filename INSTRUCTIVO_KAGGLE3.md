# Ejecución de las fases 5–8 desde `kaggle3`

PlUS esta guía parte de la rama `kaggle3`, con el panel de 370 clientes y los adaptadores de B ya implementados.

**Estado comprobado:** el pull trajo 46 pruebas aprobadas y artefactos `-smoke` de 8 clientes × 1 origen × 24 horas. No trajo los resultados completos de la fase 4. Las fases 5–7 pueden correrse ahora; la comparación final de fase 8 requiere recuperar de A los ocho experimentos principales o, como último recurso, ejecutar solo los que falten. **No ejecutes la fase 4 completa de 40 configuraciones.**

## 1. Publicar exactamente este código

Los cambios de esta guía hacen que la fase 5 guarde cada configuración, la fase 6 acumule su resumen y la fase 7 conserve avances. Un clon antiguo de Kaggle no los tendrá. En tu computadora, publica `kaggle3` o sube un ZIP del código actualizado; comprueba que `INSTRUCTIVO_KAGGLE3.md` y `requirements-kaggle.txt` aparezcan en el origen que usarás en Kaggle.

Si publicas mediante Git, ejecuta estos comandos en la terminal local desde la raíz del proyecto y revisa el commit antes de subirlo:

```bash
git status --short
git add README.md requirements-kaggle.txt INSTRUCTIVO_KAGGLE3.md src/experiments/phase4.py src/experiments/phase5.py src/experiments/phase6.py src/experiments/phase7.py src/experiments/phase8.py
git commit -m "Prepara ejecucion por etapas en Kaggle"
git push origin kaggle3
```

## 2. Preparar el Notebook de Kaggle

1. Crea un Notebook nuevo y elige GPU e Internet en **Settings**.
2. Añade como **Input** tu Dataset privado que contiene `hourly.parquet` y `client_ids.csv` del mismo panel; `dataset_audit.json` es recomendable.
3. Ejecuta la siguiente celda para clonar **kaggle3**; no uses `main` ni un clon anterior sin actualizar.

```python
!git clone --branch kaggle3 --single-branch https://github.com/tsc221645/deep_learning_lab_9.git /kaggle/working/deep_learning_lab_9
%cd /kaggle/working/deep_learning_lab_9
!git branch --show-current
!git rev-parse HEAD
!python --version
!nvidia-smi
```

Si ya existe la carpeta, usa `%cd /kaggle/working/deep_learning_lab_9`, `!git fetch origin kaggle3` y `!git switch kaggle3`, seguidos de `!git pull`; no mezcles resultados de commits distintos sin registrarlos.

Instala el par compatible de PyTorch y después las dependencias de Kaggle.

```python
%pip install -q torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
%pip install -q -r requirements-kaggle.txt
```

Reinicia **solo el kernel** después de instalar, vuelve a la carpeta del proyecto y comprueba que CUDA y `torchvision::nms` funcionan.

```python
%cd /kaggle/working/deep_learning_lab_9
import torch, torchvision
from torchvision.ops import nms
print(torch.__version__, torchvision.__version__, torch.cuda.is_available())
assert torch.cuda.is_available()
```

Busca el panel por su nombre real en `/kaggle/input`, ya que la carpeta del Dataset puede no conservar mayúsculas y espacios del título visible.

```python
from pathlib import Path
from shutil import copy2
matches = sorted(Path('/kaggle/input').rglob('hourly.parquet'))
print('Paneles encontrados:', matches)
assert len(matches) == 1, 'Añade exactamente un Dataset que contenga hourly.parquet'
source = matches[0].parent
target = Path('/kaggle/working/deep_learning_lab_9/data/processed')
target.mkdir(parents=True, exist_ok=True)
for filename in ('hourly.parquet', 'client_ids.csv'):
    assert (source / filename).is_file(), f'Falta {source / filename}'
    copy2(source / filename, target / filename)
audit = source / 'dataset_audit.json'
if audit.is_file():
    copy2(audit, target / audit.name)
print(sorted(p.name for p in target.iterdir()))
```

Corre las pruebas porque este commit tiene cambios nuevos en los runners.

```python
!python -m pytest -q
```

## 3. Recuperar outputs anteriores, si existen

Si tienes un ZIP, Dataset o versión de Notebook con `outputs/` de la sesión anterior, añádelo como Input y copia **toda** esa carpeta al proyecto antes de seguir. No basta con `phase4_summary.csv`: la fase 8 lee también `predictions.parquet` y `experiment.json` por modelo. Si esos outputs se perdieron, **salta este paso y comienza B**; no repitas la fase 4 ahora.

```python
!find /kaggle/input -name phase4_summary.csv -o -name phase5_validation.csv
```

Después de localizar el directorio real, copia su contenido a `/kaggle/working/deep_learning_lab_9/outputs/` y verifica los archivos. No escribas dentro de `/kaggle/input`, que es de solo lectura.

## 4. Fase 5: LSTM, una configuración por celda

Haz primero el smoke en el entorno nuevo. Sus 192 predicciones por modelo no son resultados finales.

```python
!python -m src.experiments.smoke_b --models lstm_recursive lstm_multioutput lstm_seq2seq
```

La selección completa requiere nueve ajustes: tres estrategias × L de 48, 168 y 336 horas. **Ejecuta una línea por celda, espera a que termine y guarda `outputs/` después de cada ajuste importante.** Cada ajuste terminado agrega una fila a `outputs/metrics/phase5_validation.csv`; el comando omite las configuraciones ya registradas cuyo log esté presente. `{}` en la salida de una selección parcial indica que aún faltan candidatos. `phase5_selected.json` aparece al terminar los nueve.

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

Cuando exista `outputs/metrics/phase5_selected.json`, evalúa **solo las tres configuraciones elegidas en validación** con el mismo backtest de 92 orígenes.

```python
!python -m src.experiments.phase5 test --models lstm_recursive
!python -m src.experiments.phase5 test --models lstm_multioutput
!python -m src.experiments.phase5 test --models lstm_seq2seq
```

## 5. Fase 6: modelos avanzados

Comprueba primero TFT, DeepAR y Chronos con smoke. En la computadora local TFT/DeepAR habían fallado por VRAM antes de limitar los lotes; el smoke del pull pasó en otra GPU, lo cual no garantiza la corrida completa en Kaggle.

```python
!python -m src.experiments.smoke_b --models tft deepar chronos
```

Ejecuta cada modelo completo por separado. `phase6_summary.csv` se acumula por `config_id`, y los pronósticos se guardan en `outputs/predictions/`. Chronos descarga pesos si no están en caché.

```python
!python -m src.experiments.phase6 --models NHITS
!python -m src.experiments.phase6 --models TCN
!python -m src.experiments.phase6 --models PatchTST
!python -m src.experiments.phase6 --models TFT
!python -m src.experiments.phase6 --models DeepAR
!python -m src.experiments.chronos_bolt
```

Si uno falla por memoria, conserva el traceback y los outputs anteriores; no reduzcas clientes solo para ese modelo, porque se perdería la igualdad del panel entre modelos.

## 6. Fase 7: experimentos adicionales

La sensibilidad a historia se divide por familia; cada línea evalúa 3, 12 y 30 meses, guarda cada configuración terminada y omite las ya registradas en `phase7_history.csv`.

```python
!python -m src.experiments.phase7 history --history-models seasonal_naive_24
!python -m src.experiments.phase7 history --history-models lightgbm
!python -m src.experiments.phase7 history --history-models dlinear
!python -m src.experiments.phase7 history --history-models recurrent
!python -m src.experiments.phase7 history --history-models patchtst
```

Los comandos siguientes seleccionan modelos con validación y realizan reajustes mensuales, diez modelos locales y tres semillas. La selección deep guarda cada candidato terminado en `phase7_deep_candidates.json` o `phase7_own_candidates.json`; restaura esos archivos entre sesiones. Ejecuta un comando por sesión si el anterior tardó varias horas.

```python
!python -m src.experiments.phase7 monthly
!python -m src.experiments.phase7 local
!python -m src.experiments.phase7 seeds
```

## 7. Fase 8: límite actual y salida final

**No ejecutes todavía la fase 8** si no existen las predicciones completas de A. El código exige ocho experimentos principales de fase 4, además de B; los CSV `-smoke` del pull no cumplen esa condición. La primera opción es solicitar a A sus carpetas completas `outputs/predictions/` y `outputs/metrics/` y verificar que cada modelo tenga `experiment.json` y `predictions.parquet` con 370 clientes y 816 960 predicciones.

Al terminar las fases 5–7, esta celda de solo lectura enumera los experimentos que aún faltan antes de lanzar fase 8:

```python
from pathlib import Path
from src.experiments.phase8 import expected_runs, _slug
for model, config_id in expected_runs(Path('outputs')).items():
    folder = Path('outputs/predictions') / _slug(model) / config_id
    print(model, (folder / 'experiment.json').exists(),
          (folder / 'predictions.parquet').exists())
```

Si A confirma que esos resultados no existen, habrá que decidir una corrida posterior de **solo los modelos principales faltantes** mediante `phase4 --primary-only`; AutoETS puede consumir muchas horas de CPU. Esta guía no incluye la fase 4 en la ruta inmediata de B ni presenta esa corrida como un paso rápido.

Cuando estén todos los artefactos de A y B, ejecuta:

```python
!python -m src.experiments.phase8 --repo-url https://github.com/tsc221645/deep_learning_lab_9
```

La fase 8 genera tabla, figuras y `report/informe_borrador.md`; no produce el PDF final. El PDF debe redactarse con los resultados reales, responder las preguntas de la guía, mostrar el enlace Git al final y tener máximo cinco páginas. El notebook unificado también debe revisarse antes de entregar.

## 8. Evitar otra pérdida de resultados

**No dependas de que el indicador “Output” de Kaggle aumente en vivo.** Al terminar un ajuste, verifica los archivos reales:

```python
!du -sh outputs
!find outputs -name experiment.json | wc -l
!ls -lh outputs/metrics
```

Antes de cerrar una sesión, crea un ZIP de `outputs/`, descárgalo desde el explorador de archivos de Kaggle o mediante el enlace mostrado, y comprueba que el ZIP exista fuera de la sesión. También puedes guardar outputs como Dataset/versión de Notebook y añadirlos como Input de la siguiente sesión, pero **verifica que los archivos aparezcan** antes de dar por guardado el avance.

```python
!zip -qr /kaggle/working/resultados_actuales.zip outputs
from IPython.display import FileLink
FileLink('/kaggle/working/resultados_actuales.zip')
```

En la siguiente sesión, sube ese ZIP como Dataset privado, añádelo como Input y restaura `outputs/` antes de continuar. `Save & Run All` inicia una sesión limpia y vuelve a ejecutar todas las celdas; conviene mantener un Notebook corto por etapa. Kaggle documenta hasta 12 horas por sesión CPU/GPU y hasta 20 GB de salida persistida, así que divide las corridas largas.

Fuentes: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) y [versiones compatibles de PyTorch](https://pytorch.org/get-started/previous-versions/).
