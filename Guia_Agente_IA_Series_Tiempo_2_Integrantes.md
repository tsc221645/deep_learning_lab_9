# Instrucciones para agente de IA: Laboratorio de Series de Tiempo y Forecasting

**Curso:** CC3092, Deep Learning y Sistemas Inteligentes  
**Actividad según el documento:** Laboratorio #8, Series de tiempo y forecasting  
**Entrega:** domingo 1 de noviembre de 2026 a las 23:59  
**Modalidad:** pareja, dos integrantes  
**Fuente de verdad:** `CC3092_Laboratorio9_Series_de_Tiempo-1.pdf` (el nombre del archivo indica «Laboratorio9», pero el contenido dice «Laboratorio #8»).  
**Propósito de este documento:** instrucciones operativas para un agente de programación (Codex, Claude Code u otro), organizadas por fases, entregables verificables y responsables. No sustituye el enunciado original.

---

## 0. Instrucción maestra para el agente

Actúa como **ingeniero senior en deep learning, forecasting, PyTorch, evaluación temporal y reproducibilidad científica**. Implementa de principio a fin el laboratorio especificado en el PDF. Prioriza la **corrección metodológica** sobre obtener artificialmente métricas bajas. No inventes resultados, gráficas, tiempos de entrenamiento, bibliografía, ni afirmes haber ejecutado experimentos que no corriste. Si una biblioteca presenta incompatibilidades, documenta el problema, versión y solución; preserva el objetivo del laboratorio. Trabaja por fases y valida los criterios de aceptación de cada fase antes de continuar.

**Reglas no negociables:**

1. Usa exactamente el mismo conjunto de clientes para todos los modelos (370 si es viable; si no, mínimo 100 seleccionados con semilla fija).
2. Todos los pronósticos son **day-ahead** de **24 horas**, emitidos a las 00:00 con información observada solamente hasta las 23:00 del día anterior. No utilices información real del horizonte en las entradas.
3. Respeta los cortes cronológicos: entrenamiento **2012-01-01 a 2014-06-30**, validación **2014-07-01 a 2014-09-30**, prueba **2014-10-01 a 2014-12-31**. Para el ajuste final previo a prueba se pueden utilizar entrenamiento + validación **solo después de seleccionar hiperparámetros**, sin mirar resultados de prueba.
4. Evalúa con el **mismo motor `backtest(modelo)`**, los mismos 92 orígenes diarios, clientes, fechas, métricas y convenciones para todas las familias, incluidos los modelos de librería.
5. Calcula métricas en la **escala física original de kWh**, no sobre series normalizadas. Todos los escaladores se ajustan únicamente con el tramo de entrenamiento disponible en cada ajuste.
6. Guarda **predicciones verificables**, métricas y metadatos; ninguna conclusión debe existir sin un experimento rastreable.
7. Implementa todas las familias, los experimentos adicionales, las comparaciones y el reporte indicados en el PDF; no reemplaces modelos difíciles por otros sin documentarlo.
8. El informe final será un **PDF de máximo 5 páginas** y debe incluir el enlace al repositorio Git. El repositorio contendrá un **Jupyter Notebook completo y comentado**.

> **Nota técnica que debe validarse con el dataset:** los datos originales se describen como potencia en kW a intervalos de 15 minutos y el objetivo como energía horaria en kWh. Si cada registro es una potencia media del intervalo, calcula `kWh_hora = sum(kW_15min * 0.25 h)`; **no sumes cuatro valores de kW como si fueran kWh**. Confirma la semántica del repositorio UCI y documenta la decisión. Si el docente interpreta la agregación literalmente de otra manera, explica la discrepancia y mantenla consistente en todos los modelos.

---

## 1. División consecutiva y equilibrada: A termina, después B empieza

**EJECUCIÓN ESTRICTAMENTE SECUENCIAL. NO INTERCALAR FASES. NO PROGRAMAR EN PARALELO.** El **Integrante A realiza y termina las fases 1, 2, 3 y 4**. Únicamente cuando el código, pruebas, resultados y documentación de esas fases estén completos y publicados en la rama principal, el **Integrante B comienza las fases 5, 6, 7 y 8**, trabajando sobre la base entregada. A no necesita volver a programar durante el bloque de B: los contratos se definen e implementan completos antes del traspaso.

| Orden | Responsable único de ejecución | Fases | Responsabilidad y entregable final |
|---|---|---|---|
| **Primero** | **Integrante A** | **1 → 2 → 3 → 4** | Entorno, datos, exploración, calendarios, Dataset de PyTorch, auditoría de fugas, investigación transversal, evaluador común y métricas; baselines estadísticos, LightGBM, MLP, DLinear, NLinear y sus experimentos iniciales. Entrega una base **funcional y evaluada**. |
| **Después** | **Integrante B** | **5 → 6 → 7 → 8** | LSTM y sus tres estrategias, seq2seq, N-HiTS/N-BEATS, TCN, PatchTST, TFT/iTransformer, DeepAR, Chronos-Bolt; experimentos restantes, 10 modelos locales, 3 semillas, reentrenamiento, comparativas y reporte final. |

**Equilibrio:** A asume toda la ingeniería de datos y evaluación, más los modelos clásicos y lineales; B recibe esta infraestructura ya probada y asume los modelos de mayor complejidad, experimentos comparativos e integración final. Las dos mitades son de trabajo sustancial, aunque la duración de entrenamientos puede variar según el hardware. **No debe imponerse al integrante B reconstruir el pipeline de A.**

**Hito de traspaso A → B (obligatorio):** repositorio en estado limpio; dependencias fijadas; datos procesados reproducibles; IDs de clientes y splits congelados; `Dataset` y `backtest(modelo)` probados; modelos de A ejecutados con resultados y pronósticos guardados; notebook con las secciones de A; documentación de interfaces, comandos y decisiones (DST, unidades, normalización, MASE); checklist de pruebas aprobado. Si algo falta, A lo completa antes de entregar. Después B continúa **sin pedir contribuciones paralelas a A**, salvo dudas puntuales de comprensión.

---

## 2. Organización recomendada del repositorio

```text
forecasting-lab/
├── README.md
├── requirements.txt                 # o pyproject.toml, versiones fijadas
├── .gitignore
├── configs/
│   ├── base.yaml
│   └── experiments.yaml
├── data/
│   ├── raw/                         # ignorado por Git
│   └── processed/                   # ignorado por Git
├── src/
│   ├── data/
│   │   ├── download.py              # A
│   │   ├── preprocess.py            # A
│   │   ├── calendars.py             # A
│   │   ├── windows.py               # A
│   │   └── splits.py                # A
│   ├── evaluation/
│   │   ├── backtest.py               # A
│   │   ├── metrics.py                # A
│   │   └── plots.py                  # B
│   ├── models/
│   │   ├── baselines.py              # A
│   │   ├── lightgbm_model.py         # A
│   │   ├── mlp.py                    # A
│   │   ├── dlinear.py                # A
│   │   ├── nlinear.py                # A
│   │   ├── recurrent.py              # B
│   │   ├── seq2seq.py                # B
│   │   ├── neuralforecast_models.py  # B
│   │   ├── deepar.py                 # B
│   │   └── chronos.py                # B
│   ├── training/
│   │   ├── trainer.py                # A
│   │   ├── logging.py                # A
│   │   └── scaling.py                # A
│   └── utils/
│       ├── seed.py                   # A
│       └── profiling.py              # B
├── tests/
│   ├── test_times_and_leakage.py     # A
│   ├── test_dataset_shapes.py        # B
│   ├── test_backtest.py              # A
│   └── test_metrics.py               # B
├── notebooks/
│   └── laboratorio_series_tiempo.ipynb
├── outputs/
│   ├── metrics/
│   ├── predictions/
│   ├── experiments/
│   └── figures/
└── report/
    ├── informe.md
    └── informe.pdf
```

El notebook puede importar módulos de `src/` para evitar duplicación, pero debe poder reproducir y **mostrar de manera entendible** la carga, EDA, Dataset, backtest, modelos, iteraciones, métricas y gráficas. No almacenar credenciales, archivos temporales grandes, checkpoints innecesarios ni el dataset bruto en Git.

### Contratos mínimos entre módulos

- **Panel procesado**: estructura tabular consistente con `unique_id`, `ds` (timestamp horario) y `y` (kWh), más covariables de calendario, sin huecos no justificados ni duplicados `unique_id, ds`.
- **Dataset PyTorch**: `X.shape = (batch, L, features)`; `y.shape = (batch, H)`; incluir metadatos `unique_id`, `origin`, `target_timestamps` y covariables futuras disponibles cuando el modelo las admita.
- **Interfaz común para modelos**: adaptadores con `fit(train, val, config)` y `predict(history, future_calendar, horizon=24, origin=None)`; `predict` devuelve exactamente H valores por cliente, o cuantiles adicionales cuando aplique. Los modelos zero-shot omiten `fit` sin romper el adaptador.
- **`backtest(modelo)`**: recibe un adaptador y una lista controlada de orígenes, verifica que no hay fechas posteriores al corte en entradas observadas, genera predicciones en formato largo y calcula métricas con un único módulo.
- **Predicciones exportadas**: columnas obligatorias `unique_id, ds, modelo, y_hat`; agregar `origin`, `y_true`, `seed`, `config_id`, `quantile` (si aplica) sin eliminar las requeridas. Comprobar unicidad por cliente, tiempo, modelo y configuración.
- **Registro experimental**: `model, config_id, seed, L, H, history_months, normalization, params_total, params_trainable, train_time_s, inference_time_s, gpu_peak_mb, hardware, metrics, artifacts`.

---

## 3. Desarrollo por fases, estrictamente consecutivas

### BLOQUE 1: INTEGRANTE A. Ejecutar fases 1 → 2 → 3 → 4 completas antes de entregar.

### Fase 1. Entorno, datos y acuerdos de reproducibilidad

**Responsable único: INTEGRANTE A.**

1. Crear entorno Python reproducible con versiones instalables de `torch`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `statsforecast`, `neuralforecast`, `lightgbm`, `chronos-forecasting`, bibliotecas para feriados y pruebas. Comprobar versiones reales y compatibilidades antes de fijarlas.
2. Crear repositorio, estructura de directorios, CLI o scripts de reproducción, README y semillas (Python, NumPy, PyTorch, bibliotecas aplicables).
3. Descargar `ElectricityLoadDiagrams20112014` de UCI desde la URL del enunciado; extraer `LD2011_2014.txt` y cargar con separador `;`, decimal `,` e índice temporal.
4. Auditar frecuencias, valores ausentes, duplicados, valores no positivos, fechas disponibles por cliente y naturaleza de los ceros. Determinar clientes con datos completos a partir de 2012-01-01 y criterios de exclusión sin usar información futura para normalizar.
5. Si no es viable procesar los 370, elegir **al menos 100** con generador aleatorio y semilla fija; guardar IDs escogidos. Asegurar igualdad entre modelos.
6. Definir política reproducible para cambios de horario de Portugal (dos transiciones por año): examinar timestamps originales, decidir uso de hora local vs calendario regular, manejar horas omitidas/repetidas y verificar **24 objetivos por día** en todas las predicciones. No introducir datos de futuro al imputar. Documentar cualquier decisión que altera días DST.
7. Convertir al objetivo horario `y` en kWh, con control de unidad como se advierte arriba. Validar conteos antes/después y registrar estadísticas.

**Criterios de aceptación:** descarga reproducible; panel horario sin timestamps duplicados; listado fijo de clientes; documentación clara del tratamiento de DST y unidades; conteos de clientes, horas por cliente y total del panel derivados del archivo real, no supuestos.

### Fase 2. Exploración y características de calendario

**Responsable único: INTEGRANTE A.**

**A:**

1. Graficar consumo promedio por hora del día y día de la semana; graficar variaciones por cliente si contribuyen a la interpretación.
2. Señalar feriados nacionales de Portugal y visualizar Navidad y Año Nuevo (incluyendo semana del 22 al 28 de diciembre).
3. Calcular mínimo, mediana y máximo de **consumo promedio por cliente**, además de distribución de escalas. Explicar por qué conviene normalización global, por serie o por ventana.
4. Implementar normalización escogida y probar su inversión exacta antes de calcular métricas. Si se usa z-score por serie, estadísticos exclusivamente del entrenamiento pertinente.

**A, continuación:**

1. Crear covariables conocidas de antemano: hora, día de semana, mes y feriado; comparar codificación one-hot, cíclica seno/coseno o embeddings y justificar selección.
2. Implementar `torch.utils.data.Dataset` que construya ventanas deslizantes con L=168, H=24 y stride de 1 hora. Asegurar shapes `X=(batch,L,features)` y `y=(batch,H)`; para seq2seq, suministrar el calendario del futuro como tensor **separado**.
3. Separar ventanas temporalmente por **fecha de objetivo**. En validación/prueba se admite usar historia anterior al corte para construir X; **nunca** permitir targets que crucen hacia el siguiente conjunto por accidente.
4. Reportar cantidad de ventanas de entrenamiento, validación y prueba con la regla de elegibilidad precisa. Distinguir ventanas horarias stride=1 para entrenamiento del backtesting diario stride=24 en prueba; no confundir sus conteos.
5. Implementar pruebas automáticas de las cuatro fugas exigidas: a) splits aleatorios de ventanas; b) escalador ajustado con toda la serie; c) early stopping con test; d) covariables temporalmente desalineadas. Mostrar con código cómo se bloquean.

**Criterios de aceptación:** cada ventana tiene origen y targets rastreables; tests de fuga ejecutados; escalado reversible; gráficos de EDA y conteos reales guardados.

### Fase 3. Investigación técnica y evaluador común

**INTEGRANTE A investiga y documenta:** `nn.LSTM`, `nn.GRU` (incluyendo `output`, `h_n`, `c_n`, `input_size`, `hidden_size`, `num_layers`, `batch_first`, `dropout`), StatsForecast y NeuralForecast (esquema `unique_id, ds, y`, `cross_validation`, `n_windows`, `step_size`, `refit`); modelo local vs global; MASE vs WAPE.

**El mismo integrante A investiga también:** `nn.Conv1d` con dilatación y padding causal (cálculo del campo receptivo TCN), `nn.TransformerEncoderLayer` y codificación posicional, `BaseChronosPipeline.predict_quantiles` con `prediction_length` y `quantile_levels`; teacher forcing, RevIN, pinball loss y cuantiles.

Para cada API, registrar propósito, firma y parámetros importantes con ejemplos mínimos compatibles con la versión instalada. Explicar diferencias entre conceptos y el impacto en este laboratorio. Agregar referencias a documentación oficial en notas del notebook/informe. **No basta nombrar conceptos; demostrar comprensión y elección de diseño.**


**Continuación de la fase 3: implementar y probar la evaluación antes de desarrollar modelos.**

1. Construir `backtest(modelo)` para 92 orígenes diarios del **1 de octubre al 31 de diciembre de 2014**, con 24 horas pronosticadas por origen para cada cliente.
2. A cada origen proporcionar únicamente consumo observado hasta las 23:00 del día anterior y calendario conocido de los próximos 24 pasos. El modelo no debe recibir `y_true` del futuro aunque su interfaz lo permita técnicamente.
3. Ajustar pesos **una vez** con información disponible hasta 2014-09-30 para la prueba principal. Hiperparámetros y early stopping solo en validación. Registrar explícitamente si un modelo usa ajuste final sobre train + validación y cómo se determina el número de epochs.
4. Métricas agregadas sobre todos los clientes y orígenes: MAE, RMSE, WAPE = `sum(abs(error))/sum(abs(y_true))`, MAPE con regla explícita para denominadores cercanos a cero y MASE.
5. Para **MASE**, definir y documentar la referencia estacional (por ejemplo, 24 horas) y calcular denominadores solo con el tramo de entrenamiento permitido, sin observar el período de prueba. Si el denominador es cero, marcar la métrica de la serie como no definida y explicar agregación; no sustituirlo silenciosamente por epsilon arbitrario.
6. Guardar MAE y RMSE **por cliente**, error por paso h=1..24, por hora del día y para la semana 22–28 de diciembre. Las métricas probabilísticas se calculan donde corresponda.
7. Registrar tiempo de entrenamiento, tiempo total de inferencia de los 92 orígenes, parámetros totales/entrenables y pico de memoria GPU (`N/A` cuando no hay CUDA; nunca sustituir con 0 como si fuera medición).
8. Crear tests con un predictor artificial cuya salida se pueda verificar exactamente; comprobar 92×24×N predicciones, ausencia de fugas, dimensiones, orden de los timestamps y métricas conocidas.

**Atención:** el enunciado pide, además, **reentrenamiento al inicio de cada mes** para el mejor deep learning y el mejor baseline, con ventana expandible (tres reajustes mensuales). Es una evaluación **separada** de la principal: en cada ajuste permitir solo lo ocurrido antes del nuevo mes y comparar el cambio de error; evitar usar los targets del mismo mes antes de que ocurran.

### Fase 4. Baselines, LightGBM y redes lineales (fin del bloque A)

**Responsable único: INTEGRANTE A.**

**Implementaciones y entregables obligatorios:**

1. **Baselines (StatsForecast):** Naive, seasonal naive con período 24, seasonal naive con período 168 y **MSTL o AutoETS**. Registrar explícitamente cuál alternativa se eligió y por qué.
2. **LightGBM global:** modelos/estrategia de predicción H=24 con rezagos 1–24 y 168, calendario y tratamiento consistente de la estrategia multi-paso. Para orígenes, no construir lag features usando consumos futuros reales.
3. **MLP multi-salida (PyTorch):** entrada L histórica y características seleccionadas, salida 24 valores.
4. **DLinear:** descomposición tendencia/estacionalidad y proyección lineal temporal, sin fuga en el cálculo de medias móviles.
5. **NLinear:** normalización respecto al último punto de la ventana y restitución de nivel, implementada correctamente para el horizonte 24.
6. Aplicar Trainer y formatos comunes, entrenar sobre el panel acordado y producir predicciones del backtest. Registrar hiperparámetros, historia, tiempo, parámetros, curvas y métricas.


**Además, A debe ejecutar y guardar:** (i) sensibilidad a historia de entrenamiento (3, 12, 30 meses) para seasonal naive, LightGBM y DLinear; (ii) tres configuraciones de L=48, 168 y 336 donde sean compatibles; (iii) comparativa de normalización sobre modelos propios; (iv) curvas y logs con pérdida train/validación, MAE/MASE por época, parámetros, tiempos y GPU. No hace falta ejecutar variantes exclusivas de modelos que B aún no ha implementado. **Guardar predicciones completas de prueba para cada modelo de A** y métricas uniformes.

**Cierre de A:** correr tests, fijar la especificación de las interfaces, escribir instrucciones de ejecución, incorporar su notebook y dejar todo integrado en `main`. Crear `HANDOFF_A_B.md` que explique cómo reutilizar datos, modelos, configuraciones, `backtest` y outputs sin tocar decisiones metodológicas. **Aquí termina por completo el trabajo de A.**

---

### BLOQUE 2: INTEGRANTE B. Ejecutar fases 5 → 6 → 7 → 8 después del traspaso.

### Fase 5. Recurrentes y estrategias multi-paso (inicio del bloque B)

**Responsable único: INTEGRANTE B. Prerrequisito: fase 4 cerrada.**

1. Leer `HANDOFF_A_B.md`, ejecutar los tests existentes y verificar el pipeline y las predicciones de referencia sin reconstruir la infraestructura.
2. Implementar **LSTM propio con tres modalidades**: autorregresiva un paso a la vez, multi-salida con horizonte H=24 y **seq2seq encoder–decoder** con covariables futuras de calendario. Teacher forcing solo durante entrenamiento; prohibido usar objetivos futuros en inferencia.
3. Entrenar y evaluar las tres modalidades con el evaluador heredado, guardar pronósticos y comparar MAE para cada paso de h=1..24, observando la acumulación de error.
4. Ajustar hiperparámetros de redes recurrentes únicamente en validación. Documentar arquitectura, parámetros, capas, dropout, batch, learning rate, tiempos y hardware.
5. No modificar los resultados de A ni cambiar la selección de clientes o el protocolo de backtesting. Si encuentra un error de infraestructura, documentarlo, corregirlo con prueba de regresión y volver a ejecutar las evaluaciones afectadas antes de comparar.

---

### Fase 6. Modelos avanzados, probabilísticos y fundacionales

**Responsable único: INTEGRANTE B.**

1. Implementar **N-HiTS o N-BEATS** y **TCN** utilizando NeuralForecast.
2. Implementar **PatchTST** y uno entre **TFT** e **iTransformer**; indicar cuál opción se escogió y por qué.
3. Implementar **DeepAR**, extraer cuantiles P10 y P90, calcular pinball loss y cobertura del intervalo.
4. Evaluar **Chronos-Bolt zero-shot**, sin ajuste de pesos ni fine-tuning, usando el mismo contexto y horizonte; medir tiempo de inferencia. Utilizar `BaseChronosPipeline.predict_quantiles` cuando la versión lo soporte.
5. Encapsular todos los modelos detrás del adaptador de pronóstico ya definido por A, respetar el backtest único y exportar predicciones `unique_id, ds, modelo, y_hat`.
6. Registrar configuraciones, parámetros totales/entrenables cuando apliquen, tiempos, hardware, métricas por cliente/hora/paso y diagnósticos de fallos.

---

### Fase 7. Experimentos, variaciones y reentrenamiento

**Responsable único: INTEGRANTE B.**

1. Ejecutar una iteración que compare al menos **L = 48, 168 y 336 horas** en modelos compatibles. Cambiar únicamente L cuando se estudie su impacto.
2. Ejecutar otra iteración comparando al menos **3, 12 y 30 meses de historia** de entrenamiento en modelos compatibles; mantener la ventana final de validación fija y distinguir los cortes cronológicos reales de cada configuración.
3. Evaluar selectivamente `normalization`, `hidden_size`, `num_layers`, `dropout`, `learning_rate`, `batch_size`. No es necesario un grid exhaustivo si el cómputo no lo permite, pero debe haber **al menos 3 iteraciones graficadas** y una justificación de las pruebas realizadas.
4. En **cada epoch** registrar training loss, validation loss, validación MAE y MASE, tiempo por epoch. Para cada iteración registrar parámetros totales/entrenables, tiempo total, tiempo de inferencia de 92 orígenes, uso de memoria GPU y hardware.
5. Ajustar learning rate/early stopping **solo con validación**. Prueba una vez por configuración seleccionada, sin usar la prueba como herramienta de tuning.
6. Comparar curvas de pérdida de al menos tres iteraciones y responder si menor training loss asegura mejor MASE de validación.
7. Experimento local/global: entrenar el mejor modelo propio global y **10 modelos locales** (uno por cada cliente escogido de forma reproducible). Comparar MASE sobre exactamente esos diez clientes con las mismas fechas.
8. Comparar las tres estrategias LSTM con una gráfica MAE vs paso del horizonte. Analizar acumulación de error autorregresivo.
9. Ejecutar mejor modelo con **3 semillas** y reportar desviación estándar del MASE.

**Gráfica de historia exigida:** MASE vs meses de entrenamiento para **seasonal naive, LightGBM, DLinear, mejor recurrente y PatchTST**, con **Chronos como línea de referencia**. Usar los resultados ya entregados por A para sus familias y correr los experimentos correspondientes a las familias de B. Aclarar cuando modelos como seasonal naive o Chronos no tienen entrenamiento propiamente dicho y cómo se definió el contraste con la historia disponible.


**Reentrenamiento mensual y cuantiles, a cargo de B:**

1. Seleccionar el mejor baseline y mejor modelo deep learning con criterio previamente fijado, **sin usar resultados de test para sintonizar hiperparámetros**. Para el análisis comparativo posterior, indicar explícitamente que la elección final del ganador por test es descriptiva, no optimización de test.
2. Repetir las predicciones de octubre, noviembre y diciembre reajustando modelos a principios de cada mes con **ventana expandible** y solo datos anteriores al mes, permitidos por el protocolo.
3. Guardar métricas por mes y error agregado del esquema con reentrenamiento; comparar con ajuste único.
4. Para DeepAR y todo otro modelo probabilístico que efectivamente emita cuantiles, calcular pinball loss y cobertura de P10–P90 en la escala original. Comprobar `P10 <= P90`, definición de cuantiles y consistencia temporal.
5. Documentar si la implementación de Chronos elegida permite `predict_quantiles`; si la API varía con la versión, adaptar el código y describir cómo se obtienen P10/P90 sin inventarlos.

### Fase 8. Comparación, interpretación y reporte

**Responsable único: INTEGRANTE B.** A dejó su metodología y resultados documentados en el traspaso; B los utiliza para la comparación, genera las figuras finales y prepara la entrega.

Generar una **tabla por modelo**, ordenada de menor a mayor **MASE de prueba**, con: familia, L, estrategia multi-paso, uso de covariables, parámetros totales/entrenables, tiempo de entrenamiento, inferencia y hardware, MAE, RMSE, WAPE, MASE H=24, MASE H=168 **si aplica**, MASE de la semana del 22 al 28 de diciembre, pinball loss y cobertura P10–P90 en los probabilísticos. No rellenar métricas que no se evaluaron; señalar `N/A` y explicar.

Crear al menos las gráficas exigidas:

1. **MASE en prueba vs tiempo de entrenamiento (eje de tiempo logarítmico)**, un punto por modelo y frontera de Pareto; Chronos zero-shot no posee tiempo de entrenamiento, por lo que debe representarse aparte o excluirse del eje log dejando nota.
2. **MAE vs h = 1..24**, una curva por modelo; cuidar legibilidad y, si hace falta, agregar versión suplementaria por familias.
3. **MASE vs meses de historia**, con seasonal naive, LightGBM, DLinear, recurrente y PatchTST; Chronos como referencia.
4. Además: perfiles temporales/feriados, curvas de entrenamiento para al menos tres iteraciones, comparativa de estrategias LSTM, análisis diciembre y, si es útil, intervalos probabilísticos.

Responder con evidencia a todas estas preguntas: ganador y mejora sobre seasonal naive; coste de entrenamiento/mantenimiento vs ganancia; DLinear frente a Transformers y en qué horizonte; estrategia multi-paso superior y propagación del error; efecto de L, normalización e historia; a cuántos meses DL supera a LightGBM/estadísticos (o si no se observó); local vs global y clientes beneficiados; Chronos zero-shot vs entrenados y caso de uso; comportamiento en Navidad/Año Nuevo y feriados; sensibilidad a semillas (media y desviación estándar del MASE).

**Reporte PDF (máx. 5 páginas):** condensar investigación, preparación/auditoría de fugas, iteraciones, estrategias, resultados completos, probabilísticos, figuras y conclusiones. Agregar enlace al repositorio Git **al final del PDF**. Mantener notebooks, logs, métricas y figuras completas en el repositorio para respaldar lo que no quepa en el PDF.

---

## 4. Orden obligatorio de ejecución y colaboración

**Prohibido alternar responsables:** un agente o integrante debe completar una fase y sus pruebas antes de iniciar la siguiente. El orden **nunca** es A1, B1, A2, B2. El orden es **A1, A2, A3, A4, ENTREGA, B5, B6, B7, B8**.

1. **A:** realiza de principio a fin las fases 1–4 y genera los artefactos, métricas, predicciones, pruebas y notebook preliminar.
2. **A:** consolida todo en `main`, ejecuta las pruebas desde un entorno limpio y redacta `HANDOFF_A_B.md` con rutas, interfaces, comandos, modelos listos, outputs y decisiones metodológicas.
3. **Traspaso:** B clona o actualiza `main`, reproduce el conjunto de pruebas y lee el handoff. No se abre una segunda línea de implementación de A.
4. **B:** realiza de principio a fin las fases 5–8, reutilizando las salidas de A y ejecutando cualquier experimento cruzado aún pendiente desde el código existente.
5. **B:** produce el notebook unificado, la tabla ordenada por MASE, visualizaciones, PDF de máximo cinco páginas y enlace a Git; comprueba cobertura de los entregables.

### Política de Git y pruebas

- A trabaja en `integrante-a-fases-1-4`, integra a `main` tras completar fase 4 y crea un commit o tag `handoff-a-completo`.
- B parte exactamente de `handoff-a-completo`, trabaja en `integrante-b-fases-5-8` e integra después de finalizar fase 8.
- Toda modificación de interfaces después del traspaso exige pruebas de regresión y, si cambia pronósticos o métricas, volver a ejecutar los modelos afectados.
- Los resultados se guardan en directorios separados `outputs/<modelo>/<config>/<semilla>/`; no sobrescribir resultados previos sin dejar rastro.
- Para problemas de memoria, reducir batch o usar un subconjunto fijo de mínimo 100 clientes **para todos los modelos**, nunca distintos paneles entre A y B.

---

## 5. Matriz de cobertura: requisito del PDF → evidencia → dueño

| Requisito | Evidencia comprobable | Dueño |
|---|---|---|
| UCI, panel completo/100+, fechas, unidades y DST | Script + auditoría + conteos | A |
| Perfil horario/semanal, feriados, Navidad, escalas | Gráficas + tabla estadística | A |
| Covariables y Dataset L168/H24/stride1 | Código + shapes + conteos | A |
| Cuatro fugas: split, escalado, early stopping, alineación | 4 tests + demostraciones | A |
| Investigación de APIs y conceptos | Sección técnica con ejemplos/citas | A |
| Backtest 92 orígenes y métricas comunes | Función única + tests + archivos de salida | A |
| Naive, seasonal 24/168, MSTL/AutoETS | Código, resultados y tiempos | A |
| LightGBM, MLP, DLinear, NLinear | Código, entrenamiento y backtest | A |
| LSTM recurrente y 3 estrategias; seq2seq | 3 implementaciones y gráfica MAE/h | A |
| N-HiTS/N-BEATS, TCN | Entrenamiento y resultados | A |
| PatchTST, TFT/iTransformer | Entrenamiento y resultados | B |
| DeepAR probabilístico | Cuantiles, pinball, cobertura | B |
| Chronos-Bolt zero-shot | Pronóstico, tiempo, comparación | B |
| Iteraciones L, historia, curvas y profiling | Logs de epochs, 3 curvas y tabla | A (modelos iniciales), B (resto) |
| Mejor modelo local vs global para 10 clientes | Tabla MASE por cliente | B |
| Mejor baseline y DL: reentrenamiento mensual | 3 reajustes y comparación | B |
| Semillas x3 para mejor modelo | MASE por semilla y desviación | B |
| Tabla final ordenada MASE + 3 gráficas mínimas | Artefactos generados desde outputs | B con datos de A |
| Informe PDF ≤5 páginas + link Git + notebook comentado | Entrega final verificada | B |

---

## 6. Pruebas mínimas y criterios de calidad

Exige que estas verificaciones terminen correctamente antes de afirmar «proyecto completo»:

- [ ] Todos los clientes tienen datos válidos según política documentada desde 2012; si hay exclusiones, aparecen identificadas.
- [ ] La conversión 15 minutos → hora tiene dimensión kWh y la política DST es consistente con la definición de 24 horas por origen.
- [ ] Las ventanas tienen tamaño esperado y el calendario futuro no usa consumos reales futuros.
- [ ] Los splits son temporales, el normalizador no conoce validación/prueba cuando se ajusta y el early stopping nunca inspecciona test.
- [ ] Las 92 fechas de prueba generan 24 puntos por cliente y por modelo sin huecos o duplicados.
- [ ] Todos los modelos comparten clientes, targets, fechas y definición de MASE/WAPE.
- [ ] Las predicciones se desnormalizan antes de MAE, RMSE, WAPE, MAPE y MASE.
- [ ] Los logs incluyen training/validation loss por epoch, validation MAE/MASE, parámetros, tiempos y memoria medida o `N/A`.
- [ ] Están las tres estrategias LSTM, tres valores de L, tres longitudes de historia, 10 clientes locales, tres semillas y los tres reajustes mensuales.
- [ ] El análisis probabilístico incluye pinball y cobertura P10–P90 donde existen cuantiles.
- [ ] Están la tabla final completa, las 3 figuras obligatorias, el análisis de feriados y las respuestas de discusión.
- [ ] El notebook se ejecuta de inicio a fin bajo las dependencias fijadas, o separa claramente ejecución de entrenamiento costosa de la reconstrucción reproducible de resultados.
- [ ] El PDF no supera 5 páginas y termina con el enlace al repositorio.

---

## 7. Formato de cierre exigido al agente

Al completar **cada fase**, responde con este esquema (sin marcar tareas pendientes como terminadas):

```markdown
### Fase X: [nombre]
- Responsable: A o B (nunca ambos en una fase)
- Archivos creados o modificados: ...
- Decisiones metodológicas: ...
- Comandos ejecutados: ...
- Tests pasados/fallidos: ...
- Experimentos realmente corridos y resultados: ...
- Artefactos guardados: ...
- Bloqueos o limitaciones verificables: ...
- Siguiente fase y dependencias: ...
```

**Al finalizar todo el proyecto**, entregar un resumen con: requerimientos completos/incompletos, listado de modelos efectivamente ejecutados, mejor modelo según MASE, comparación con seasonal naive, ruta al notebook, ruta al CSV/Parquet de pronósticos, ruta al PDF, enlace Git y comandos exactos para reproducir. Nunca afirmar que está listo para entregar si faltan experimentos o el informe no está validado.

---

## 8. Consideraciones del enunciado que requieren decisiones explícitas

1. **Nombre del archivo vs número de laboratorio:** filename dice Laboratorio9, portada dice Laboratorio #8; conservar la numeración del contenido y mencionar la diferencia.
2. **Potencia vs energía:** el PDF dice registros en kW y objetivo en kWh, por lo que hay que justificar el factor 0.25 h en la agregación si se trata de potencia media de 15 minutos.
3. **Ventanas stride 1 vs backtest stride 24:** los conteos de Dataset y los 92 orígenes de prueba corresponden a dos procedimientos distintos, aunque usen el mismo panel.
4. **MASE:** el enunciado exige MASE pero no fija explícitamente su estacionalidad ni modo de agregación; documentar la convención elegida y aplicarla a todos los modelos. H=168 solo donde aplique, sin reemplazar silenciosamente la tarea principal H=24.
5. **Mención de tres reajustes mensuales:** interpretar como reajustes al inicio de octubre, noviembre y diciembre, con historia expandible y sin información futura.
6. **Disponibilidad computacional:** el mínimo de 100 clientes es una excepción autorizada, pero solo si se usa de forma uniforme.
7. **Opciones con «o»:** `LSTM o GRU`, `MSTL o AutoETS`, `N-HiTS o N-BEATS`, `TFT o iTransformer`; respetar las opciones. Como la subsección 5.1 exige explícitamente **LSTM**, usar LSTM para comparar estrategias aunque se pruebe también GRU.
8. **Métricas probabilísticas:** el PDF exige DeepAR y evaluación probabilística; obtener cuantiles desde modelos que lo admitan, con niveles y calibración explicados.
9. **Numeración del PDF:** el documento pasa de sección 6 a sección 8; no inventar contenido de una inexistente sección 7.

**Instrucción final:** avanza directamente con la implementación siguiendo estas fases. Pide intervención humana únicamente si falta una credencial, decisión externa o recurso esencial que no puedas obtener; ante errores técnicos ordinarios, investiga, corrige, prueba y documenta.
