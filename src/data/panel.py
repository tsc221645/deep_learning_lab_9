"""Memory-efficient readers for the processed long-format panel."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HOUR_DELTA = pd.Timedelta(1, unit="h")


@dataclass(frozen=True)
class Panel:
    """Dense client-by-time view of the processed electricity panel."""

    client_ids: tuple[str, ...]
    timestamps: pd.DatetimeIndex
    values: np.ndarray

    def __post_init__(self) -> None:
        expected_shape = (len(self.client_ids), len(self.timestamps))
        if self.values.shape != expected_shape:
            raise ValueError(
                f"Panel values have shape {self.values.shape}, expected {expected_shape}"
            )


def load_panel_matrix(panel_path: Path, client_ids_path: Path) -> Panel:
    """Load one parquet row group at a time into a compact float32 matrix."""
    client_ids = tuple(
        pd.read_csv(client_ids_path)["unique_id"].astype(str).tolist()
    )
    if not client_ids or len(set(client_ids)) != len(client_ids):
        raise ValueError("Client IDs must be present and unique")

    parquet = pq.ParquetFile(panel_path)
    if parquet.metadata.num_rows % len(client_ids):
        raise ValueError("Parquet row count is not divisible by the client count")
    total_hours = parquet.metadata.num_rows // len(client_ids)
    values = np.empty((len(client_ids), total_hours), dtype=np.float32)
    time_blocks = []
    offset = 0
    previous_timestamp = None
    id_array = np.asarray(client_ids, dtype=object)[:, None]

    for row_group_index in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(
            row_group_index, columns=["unique_id", "ds", "y"]
        ).to_pandas()
        if len(frame) % len(client_ids):
            raise ValueError(
                f"Parquet row group {row_group_index} splits client records"
            )
        hours = len(frame) // len(client_ids)
        if not hours:
            raise ValueError(f"Parquet row group {row_group_index} is empty")

        id_block = frame["unique_id"].astype(str).to_numpy().reshape(
            len(client_ids), hours
        )
        if not np.array_equal(id_block, np.broadcast_to(id_array, id_block.shape)):
            raise ValueError(
                f"Client ordering differs in parquet row group {row_group_index}"
            )
        timestamp_block = pd.DatetimeIndex(
            frame["ds"].to_numpy().reshape(len(client_ids), hours)[0]
        )
        if len(timestamp_block) > 1 and not timestamp_block.to_series().diff().iloc[
            1:
        ].eq(HOUR_DELTA).all():
            raise ValueError(
                f"Parquet row group {row_group_index} has a non-hourly time grid"
            )
        all_timestamps = frame["ds"].to_numpy().reshape(len(client_ids), hours)
        if not np.array_equal(
            all_timestamps,
            np.broadcast_to(timestamp_block.to_numpy()[None, :], all_timestamps.shape),
        ):
            raise ValueError(
                f"Clients have different timestamps in row group {row_group_index}"
            )
        if previous_timestamp is not None and (
            timestamp_block[0] - previous_timestamp != HOUR_DELTA
        ):
            raise ValueError("Hourly panel has a gap or duplicate between row groups")

        next_offset = offset + hours
        values[:, offset:next_offset] = frame["y"].to_numpy(
            dtype=np.float32
        ).reshape(len(client_ids), hours)
        time_blocks.append(timestamp_block.to_numpy())
        offset = next_offset
        previous_timestamp = timestamp_block[-1]

    if offset != total_hours:
        raise ValueError(f"Loaded {offset} hours, expected {total_hours}")
    timestamps = pd.DatetimeIndex(np.concatenate(time_blocks))
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Hourly timestamps must be unique and increasing")
    if not np.isfinite(values).all():
        raise ValueError("Processed panel contains non-finite target values")

    return Panel(client_ids=client_ids, timestamps=timestamps, values=values)
