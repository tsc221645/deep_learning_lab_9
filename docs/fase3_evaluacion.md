# Fase 3: evaluación y APIs

## Contrato del evaluador

Cada adaptador implementa `fit(train, val, config)` y `predict(history, future_calendar, horizon=24, origin=None)`. Los paneles de ajuste contienen IDs, timestamps y valores, con forma `clientes x horas`. La predicción devuelve una matriz `(clientes, horizonte)`. Los modelos zero-shot pueden omitir `fit`.

`backtest` ajusta una vez y recorre los 92 orígenes diarios entre el 1 de octubre y el 31 de diciembre de 2014. En cada origen, `history` termina a las 23:00 del día anterior y `future_calendar` contiene solo las covariables conocidas del horizonte. La función llama a `predict` antes de leer los targets de ese horizonte. Valida forma, finitud, orden y unicidad de las predicciones. La historia de ajuste se limita a los últimos `history_months`; el valor predeterminado es 30.

Para el ajuste principal se puede usar validación para seleccionar hiperparámetros y época. Una vez seleccionados, `refit_train_validation=True` combina entrenamiento y validación y requiere `best_epoch`; en ese modo se entrena con un número fijo de épocas, sin early stopping sobre prueba. La opción predeterminada mantiene entrenamiento y validación separados.

`monthly_refit_backtest` es una evaluación independiente: crea un modelo nuevo al inicio de cada mes de prueba y usa solo datos anteriores a ese mes. Los hiperparámetros y `best_epoch` se mantienen fijos. El pronóstico de un día sí puede usar observaciones de días previos ya ocurridos.

Con 370 clientes, la prueba principal produce `92 x 24 x 370 = 816,960` predicciones. El Parquet de predicciones conserva `unique_id, ds, model, y_hat, origin, y_true, horizon_step, seed, config_id` y la escala MASE usada. Se guardan también métricas agregadas, por cliente, paso del horizonte, hora y fecha del 22 al 28 de diciembre, junto con un registro del experimento. `output_dir=None` permite probar sin escribir archivos.

## Métricas

- MAE y RMSE se calculan en kWh.
- WAPE es `100 * sum(abs(y - y_hat)) / sum(abs(y))`, expresado como porcentaje. Si la suma de targets es cero, queda indefinido.
- MAPE también se expresa como porcentaje. Se excluyen observaciones con `abs(y) <= 1e-8` y se informa cuántas se excluyeron.
- MASE usa estacionalidad horaria `m=24`: para cada cliente, el denominador es la media de `abs(y[t] - y[t-24])` en el tramo de entrenamiento del ajuste. Se promedian los MASE por cliente y período de ajuste con denominador positivo. Los denominadores cero quedan indefinidos y se cuentan, sin añadir un epsilon.

WAPE resume el error total y da más peso a clientes con mayor consumo. MASE escala cada serie por su error estacional de entrenamiento, lo que facilita comparar series de escalas distintas. Se reportan ambas para no esconder la diferencia. Los promedios no sustituyen las métricas por cliente.

## APIs revisadas

Las firmas se comprobaron con StatsForecast 2.1.1, NeuralForecast 3.3.0 y Chronos Forecasting 2.3.2. El entorno CUDA actual usa PyTorch 2.11.0+cu128; la versión exacta de PyTorch indicada en un registro de ejecución debe prevalecer si se cambia el entorno.

### LSTM y GRU

En PyTorch, `nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=False, ...)` consume una secuencia y devuelve `output, (h_n, c_n)`. `input_size` es el número de canales de entrada; `hidden_size` es el ancho del estado; `num_layers` apila capas. Con `batch_first=True`, `x` y `output` tienen forma `(batch, tiempo, canales)`. `h_n` y `c_n` conservan la forma `(capas x direcciones, batch, estado)`; `batch_first` no cambia esos estados. El `dropout` interno se aplica entre capas recurrentes, no después de la última.

```python
lstm = nn.LSTM(7, 64, num_layers=2, batch_first=True, dropout=0.1)
output, (h_n, c_n) = lstm(torch.zeros(32, 168, 7))
```

`nn.GRU` usa las mismas convenciones de entrada, capas y `batch_first`, pero retorna `output, h_n`; no tiene estado de celda `c_n`. Ambos pueden resumir las 168 horas observadas antes de proyectar las 24 horas siguientes. No se debe usar una RNN bidireccional sobre datos que incluyan el horizonte objetivo.

```python
gru = nn.GRU(7, 64, num_layers=2, batch_first=True, dropout=0.1)
output, h_n = gru(torch.zeros(32, 168, 7))
```

### Convolución causal y Transformer

`nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, ...)` espera `(batch, canales, tiempo)`. El padding usual es simétrico y puede dejar que una salida use valores posteriores del mismo segmento. Para una convolución causal se aplica padding solo a la izquierda: `F.pad(x, (dilation * (kernel_size - 1), 0))`, seguido de `Conv1d(..., padding=0, dilation=dilation)`. Con stride 1, el campo receptivo de capas con kernel `k_l` y dilatación `d_l` es `1 + sum((k_l - 1) * d_l)`. Para bloques con dos convoluciones por nivel, cada nivel aporta dos veces su término. Esto permite verificar si el TCN realmente cubre una semana de 168 horas.

```python
x = torch.zeros(32, 7, 168)
dilation = 2
conv = nn.Conv1d(7, 32, kernel_size=3, dilation=dilation)
y = conv(F.pad(x, (dilation * 2, 0)))
```

`nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False, norm_first=False, ...)` combina autoatención y una red feed-forward. Con `batch_first=True`, la entrada es `(batch, tiempo, d_model)`. La autoatención no codifica por sí sola el orden; se suma una codificación posicional, por ejemplo seno/coseno. El encoder puede atender a todas las posiciones de la historia observada. Un decoder que recibe tokens futuros debe usar máscara causal y no targets futuros.

```python
layer = nn.TransformerEncoderLayer(
    d_model=32, nhead=4, dim_feedforward=64, batch_first=True
)
encoded = layer(torch.zeros(32, 168, 32))
```

### StatsForecast y NeuralForecast

Ambas bibliotecas aceptan datos largos con columnas `unique_id, ds, y`. `unique_id` identifica la serie, `ds` el timestamp y `y` el consumo en kWh. StatsForecast estima muchos modelos estadísticos por serie; NeuralForecast normalmente entrena pesos compartidos entre series para una arquitectura global.

```python
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

sf = StatsForecast(
    models=[SeasonalNaive(season_length=24)], freq="h", n_jobs=1
)
sf.cross_validation(
    h=24, df=df, n_windows=92, step_size=24, refit=False
)
```

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

nf = NeuralForecast(
    models=[NHITS(h=24, input_size=168, max_steps=1)], freq="h"
)
nf.cross_validation(
    df=df, n_windows=92, step_size=24, refit=False
)
```

En ambas APIs, `n_windows` fija cuántos cortes se evalúan y `step_size` separa los cortes. `refit=True` vuelve a entrenar en cada ventana; `refit=False` reutiliza el ajuste y requiere soporte del modelo para avanzar. StatsForecast usa `refit=True` como valor predeterminado; NeuralForecast usa `False`. El evaluador propio se usa como referencia única porque controla explícitamente los 92 orígenes, el mismo conjunto de clientes, la exposición del historial y el formato de métricas para todas las familias.

### Chronos, cuantiles y teacher forcing

La interfaz de Chronos 2.3.2 es `BaseChronosPipeline.predict_quantiles(inputs, prediction_length=None, quantile_levels=[...], **kwargs)`. `inputs` puede ser un tensor `(batch, tiempo)` o una lista de series; para este laboratorio `prediction_length=24`. Devuelve cuantiles con forma `(batch, horizonte, número_de_cuantiles)` y la media con forma `(batch, horizonte)`. La media sirve como pronóstico puntual; los cuantiles describen incertidumbre y permiten evaluar cobertura o pinball loss.

```python
context = torch.zeros(370, 168)
quantiles, mean = pipeline.predict_quantiles(
    context, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9]
)
```

Teacher forcing alimenta al decoder con el target observado del paso anterior durante el entrenamiento. En inferencia ese valor no existe y se reemplaza por la predicción anterior, lo que puede acumular error. Para no filtrar información, el calendario futuro se entrega aparte y nunca se usa el consumo objetivo futuro como entrada del decoder.

RevIN normaliza cada ventana con estadísticas de su propio historial y revierte esa transformación a la salida. No aprende estadísticas con el horizonte futuro. Se diferencia del z-score por serie del laboratorio, cuyos parámetros se estiman una vez usando el tramo de entrenamiento.

```python
x = revin_layer(x, "norm")
forecast = revin_layer(forecast, "denorm")
```

Para un cuantil `q`, la pinball loss es `max(q * (y - y_hat), (q - 1) * (y - y_hat))`. Penaliza asimétricamente los errores por encima y por debajo del cuantil. Se calcula por nivel de cuantil; no se mezcla con MAE/RMSE ni se evalúa como si una sola salida puntual representara toda la distribución.

## Referencias

- [PyTorch `nn.LSTM`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html), [nn.GRU](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GRU.html), [nn.Conv1d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv1d.html) y [nn.TransformerEncoderLayer](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoderLayer.html).
- [Tutorial seq2seq de PyTorch](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), para encoder-decoder y teacher forcing.
- [StatsForecast: métodos core y cross-validation](https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html) y [NeuralForecast: cross-validation](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/cross_validation.html).
- [Chronos Forecasting](https://github.com/amazon-science/chronos-forecasting), implementación y API de `BaseChronosPipeline`.
- [RevIN: implementación oficial y artículo](https://github.com/ts-kim/RevIN).
- [Forecasting: Principles and Practice, medidas de accuracy](https://otexts.com/fpp3/accuracy.html) y [Koenker y Bassett, Regression Quantiles](https://doi.org/10.2307/1913643).
