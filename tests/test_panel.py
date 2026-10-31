import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.panel import load_panel_matrix


def test_load_panel_matrix_reads_client_time_row_groups(tmp_path):
    client_ids = ["client_a", "client_b"]
    timestamps = pd.date_range("2012-01-01", periods=3, freq="h")
    frame = pd.DataFrame(
        {
            "unique_id": np.repeat(client_ids, len(timestamps)),
            "ds": np.tile(timestamps, len(client_ids)),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    panel_path = tmp_path / "panel.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), panel_path)
    client_ids_path = tmp_path / "client_ids.csv"
    pd.DataFrame({"unique_id": client_ids}).to_csv(client_ids_path, index=False)

    panel = load_panel_matrix(panel_path, client_ids_path)

    assert panel.client_ids == tuple(client_ids)
    assert panel.timestamps.equals(timestamps)
    np.testing.assert_array_equal(panel.values, [[1, 2, 3], [4, 5, 6]])
