import numpy as np
import pandas as pd

from src.models.chronos import ChronosBoltAdapter, probabilistic_metrics


def test_quantiles_follow_frozen_client_order_not_lexicographic_order():
    adapter = ChronosBoltAdapter(context_length=2)
    adapter.client_ids = ("z", "a")
    origin = pd.Timestamp("2014-10-01")
    times = pd.date_range(origin, periods=2, freq="h")
    adapter.quantiles_by_origin[origin] = np.array(
        [[[1, 2, 3], [2, 3, 4]], [[10, 11, 12], [11, 12, 13]]]
    )
    points = pd.DataFrame(
        [
            {"unique_id": client, "ds": time, "origin": origin,
             "y_true": value, "y_hat": value}
            for client, value in (("a", 11), ("z", 2))
            for time in times
        ]
    )
    result = adapter.probabilistic_predictions(points)
    assert result.loc[result["unique_id"].eq("z"), "p50"].tolist() == [2, 3]
    assert result.loc[result["unique_id"].eq("a"), "p50"].tolist() == [11, 12]
    assert 0 <= probabilistic_metrics(result)["coverage_p10_p90"] <= 1
