# Traspaso para ejecutar el bloque del integrante B

**Estado: preparado para continuar las comprobaciones; todavía no validado para una corrida completa ni listo para entregar.** Este documento registra lo observado hasta el 31 de octubre de 2026. Las corridas largas y el informe final requieren resultados reales.

## Qué se hizo

- Se programaron las tres estrategias LSTM (recursiva, multi-salida y seq2seq), los adaptadores NeuralForecast para N-HiTS, TCN, PatchTST, TFT y DeepAR, y el adaptador zero-shot de Chronos-Bolt.
- Se añadieron comandos para selección en validación, evaluación con el backtest común, sensibilidad a historia, comparación local/global, tres semillas, reentrenamiento mensual y generación de tablas/figuras.
- Se corrigieron el nombre `AutoETS24` en la comparación final, identificadores repetidos de fase 4, la ventana expandible de los reajustes mensuales y la comparación de los valores `y_true` entre modelos. Se fijó `torchvision==0.26.0+cu128` junto con `torch==2.11.0+cu128` y se fijó el idioma portugués de los feriados.
- Después de observar falta de memoria GPU en TFT y DeepAR, se limitaron `windows_batch_size`, `inference_windows_batch_size` y `valid_batch_size` en el adaptador NeuralForecast y en el smoke. **Este último cambio aún no se ha probado.**

## Resultados observados, sin confundir smoke con resultados finales

| Comprobación | Resultado comunicado por el primer ejecutor |
|---|---|
| `python -m pytest -q` antes del último cambio de lotes | 42 pruebas pasadas en 21.70 s |
| Fase 4 `--smoke` | 8 modelos produjeron 192 predicciones cada uno, con 8 clientes y 1 origen |
| LSTM recursiva, multi-salida y seq2seq `smoke_b` | 192 predicciones por estrategia |
| N-HiTS, TCN, PatchTST `smoke_b` | 192 predicciones por modelo |
| TFT `smoke_b`, configuración anterior | `torch.OutOfMemoryError` al intentar reservar 576 MiB adicionales en GPU de 4 GiB |
| DeepAR `smoke_b`, configuración anterior | `torch.OutOfMemoryError` durante validación; intento de reserva de 29.82 GiB |
| TFT y DeepAR después de limitar lotes | **Pendiente de comprobar** |
| Chronos-Bolt `smoke_b` | **Pendiente de comprobar** |
| Fases 4–8 completas | **No ejecutadas** |

El fallo reportado fue **memoria de GPU (VRAM)**, no espacio libre en disco. Los avisos sobre `triton` y FLOPs no son la causa del fallo. Los valores de MAE/MASE de los smoke no sirven para comparar modelos: usan un solo origen y entrenamiento mínimo.

## Archivos que debe recibir la persona que ejecutará

1. **Todo el código actualizado del repositorio**, incluidos los archivos aún sin commit. Un clon de GitHub no los tendrá hasta que se confirmen y publiquen; alternativamente, enviar un ZIP del proyecto. No incluir `.venv/`, cachés ni credenciales.
2. `data/processed/hourly.parquet` y `data/processed/client_ids.csv`, juntos. El panel local existe y mide aproximadamente 85 MB; el listado fija 370 clientes. Ambos deben corresponder a la misma auditoría `data/processed/dataset_audit.json`.
3. Si no se transfiere el panel procesado, la persona tendrá que descargar y procesar `LD2011_2014.txt` con los comandos de abajo. El texto bruto local mide aproximadamente 711 MB y está ignorado por Git.
4. Los resultados completos de A para fase 4 **no están en este checkout**. La persona deberá obtenerlos de A o ejecutar `python -m src.experiments.phase4` antes de la comparación final. Los smoke no sustituyen esos resultados.

Los datos procesados y la mayoría de `outputs/` están ignorados por Git. Hay que transferirlos aparte o guardarlos como artefactos de la plataforma remota.

## Orden de ejecución para la otra persona

Ejecutar desde la raíz del proyecto, con Python y una GPU CUDA compatibles. El primer equipo usó Python 3.12, PyTorch 2.11.0+cu128 y torchvision 0.26.0+cu128; no hace falta instalar Python 3.13 solo por este proyecto. Una GPU con más VRAM que 4 GiB es recomendable para la corrida completa, pero el mínimo real sigue sin medirse.

```text
python --version
nvidia-smi
python -m pip install -r requirements.txt
python -m pytest -q
```

Si falta `data/processed/hourly.parquet`:

```text
python -m src.data.download --output-dir data/raw
python -m src.data.preprocess --input data/raw/LD2011_2014.txt --output data/processed/hourly.parquet --audit-dir data/processed --seed 42
```

Repetir las comprobaciones pequeñas **después de los cambios de tamaño de lote**. Detenerse y conservar el traceback si falla algún comando:

```text
python -m src.experiments.phase4 --smoke
python -m src.experiments.smoke_b --models lstm_recursive lstm_multioutput lstm_seq2seq
python -m src.experiments.smoke_b --models nhits tcn patchtst tft deepar
python -m src.experiments.smoke_b --models chronos
```

Chronos-Bolt puede descargar pesos públicos en su primera ejecución; se requiere acceso a Internet o una caché con esos pesos. Si los smoke pasan y están disponibles los resultados de A, las corridas completas previstas son:

```text
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

Omitir `phase4` únicamente si se reciben sus pronósticos, métricas y metadatos completos de A con los mismos IDs. Conviene dividir las corridas largas por etapa y guardar `outputs/` entre sesiones. El notebook `notebooks/laboratorio_series_tiempo.ipynb` tiene `RUN_EXPENSIVE = False` por defecto; **Run All no reemplaza estos comandos**.

## Qué falta antes de afirmar “listo para entregar”

- Confirmar de nuevo pruebas y smoke para TFT, DeepAR y Chronos tras los cambios recientes. No hay evidencia aún de que TFT/DeepAR completos quepan en la GPU de destino.
- Ejecutar las fases con todos los clientes congelados, 92 orígenes diarios y 24 horas por origen; verificar que el número de predicciones, IDs, fechas y `y_true` coincidan en todos los modelos.
- Revisar los registros por epoch de NeuralForecast: el requisito pide loss, MAE y MASE de validación y tiempo por epoch; el logger actual no acredita todavía todos esos campos.
- Revisar la selección metodológica: las variantes de historia de fase 7 y las múltiples configuraciones de fase 4 llegan a prueba; no usar sus métricas de test para escoger hiperparámetros. Documentar qué configuraciones se fijaron con validación.
- Analizar resultados, Navidad/Año Nuevo, costo, estrategias, cuantiles, comparación local/global y sensibilidad a semillas con cifras reales. `phase8` genera un **borrador**, no un informe final. Falta convertirlo en PDF de máximo 5 páginas e incluir el enlace Git real al final.

## Qué devolver al integrar resultados

Conservar y compartir `outputs/metrics/`, `outputs/predictions/`, `outputs/models/` (logs necesarios), `outputs/figures/`, `data/processed/client_ids.csv`, `data/processed/dataset_audit.json`, el panel procesado y cualquier traceback completo. Registrar también GPU, VRAM, RAM, versiones de Python y bibliotecas, y comandos efectivamente corridos. No presentar métricas de smoke como resultados finales.
