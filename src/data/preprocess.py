"""Audit the UCI quarter-hour data and convert it to hourly energy."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
QUARTER_HOURS_PER_HOUR = 4
MINUTES_PER_QUARTER = 15
FIRST_ELIGIBLE = pd.Timestamp("2012-01-01 00:00:00")
LAST_ELIGIBLE_EXCLUSIVE = pd.Timestamp("2015-01-01 00:00:00")
CHUNK_ROWS = 96 * 7
SPRING_DST_DATES = (
    "2011-03-27",
    "2012-03-25",
    "2013-03-31",
    "2014-03-30",
)
AUTUMN_DST_DATES = (
    "2011-10-30",
    "2012-10-28",
    "2013-10-27",
    "2014-10-26",
)
QUARTER_DELTA = pd.Timedelta(MINUTES_PER_QUARTER, unit="min")
HOUR_DELTA = pd.Timedelta(1, unit="h")
ZERO_DELTA = pd.Timedelta(0, unit="ns")


def read_quarter_hour_chunks(
    path: Path, chunksize: int = CHUNK_ROWS
) -> Iterator[pd.DataFrame]:
    """Read the semicolon-separated UCI file without loading it all at once."""
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        chunksize=chunksize,
        low_memory=False,
    )


def aggregate_quarter_hourly(
    timestamps: pd.DatetimeIndex, values_kw: np.ndarray
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Convert consecutive 15-minute mean kW values to hourly kWh."""
    if values_kw.ndim != 2 or len(timestamps) != values_kw.shape[0]:
        raise ValueError("Expected timestamps and a (quarter, client) value matrix")
    if len(timestamps) == 0:
        raise ValueError("Quarter-hour data must contain complete hourly groups")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Source timestamps must be unique and increasing")

    deltas = timestamps.to_series().diff().iloc[1:]
    if not deltas.eq(QUARTER_DELTA).all():
        raise ValueError("Source timestamps must be spaced exactly 15 minutes apart")

    hour_starts = timestamps.floor("h")
    group_starts = np.r_[0, np.flatnonzero(hour_starts[1:] != hour_starts[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(timestamps)]
    if not np.all(group_ends - group_starts == QUARTER_HOURS_PER_HOUR):
        raise ValueError("Each complete hour must contain exactly four readings")

    hourly_values = np.stack(
        [
            values_kw[start:end].sum(axis=0)
            for start, end in zip(group_starts, group_ends)
        ]
    ) * 0.25
    return pd.DatetimeIndex(hour_starts[group_starts]), hourly_values


def _aggregate_available_hours(
    timestamps: pd.DatetimeIndex,
    values_kw: np.ndarray,
    allow_initial_partial: bool = False,
) -> tuple[pd.DatetimeIndex, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Aggregate complete local hours and carry a partial trailing hour forward."""
    if values_kw.ndim != 2 or len(timestamps) != values_kw.shape[0]:
        raise ValueError("Expected timestamps and a (quarter, client) value matrix")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Source timestamps must be unique and increasing")
    deltas = timestamps.to_series().diff().iloc[1:]
    if not deltas.eq(QUARTER_DELTA).all():
        raise ValueError("Source timestamps must be spaced exactly 15 minutes apart")

    hour_starts = timestamps.floor("h")
    group_starts = np.r_[0, np.flatnonzero(hour_starts[1:] != hour_starts[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(timestamps)]
    trailing_partial_start = (
        group_starts[-1]
        if group_ends[-1] - group_starts[-1] < QUARTER_HOURS_PER_HOUR
        else len(timestamps)
    )
    complete_hours = []
    hourly_values = []

    for group_index, (start, end) in enumerate(zip(group_starts, group_ends)):
        if start >= trailing_partial_start:
            break
        group_timestamps = timestamps[start:end]
        if end - start != QUARTER_HOURS_PER_HOUR:
            initial_hour_is_partial = (
                allow_initial_partial
                and group_index == 0
                and group_timestamps[0] == hour_starts[start] + QUARTER_DELTA
                and hour_starts[start] < FIRST_ELIGIBLE
                and len(group_timestamps) == QUARTER_HOURS_PER_HOUR - 1
            )
            if initial_hour_is_partial:
                continue
            raise ValueError(
                f"Incomplete hour at {hour_starts[start]} has {len(group_timestamps)} readings"
            )
        expected = pd.date_range(
            hour_starts[start], periods=QUARTER_HOURS_PER_HOUR, freq="15min"
        )
        if not group_timestamps.equals(expected):
            raise ValueError(f"Quarter-hour readings are misaligned at {hour_starts[start]}")
        complete_hours.append(hour_starts[start])
        hourly_values.append(values_kw[start:end].sum(axis=0) * 0.25)

    pending_timestamps = timestamps[trailing_partial_start:]
    pending_values = values_kw[trailing_partial_start:]
    return (
        pd.DatetimeIndex(complete_hours),
        np.asarray(hourly_values, dtype=np.float64).reshape(-1, values_kw.shape[1]),
        pending_timestamps,
        pending_values,
    )


def choose_client_ids(
    client_ids: list[str], limit: int, seed: int = SEED
) -> list[str]:
    """Choose a stable, sorted client subset when a limit is requested."""
    if limit < 1:
        raise ValueError("Client limit must be at least one")
    if limit >= len(client_ids):
        return sorted(client_ids)
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.asarray(client_ids), size=limit, replace=False)
    return sorted(str(client_id) for client_id in selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_chunk(chunk: pd.DataFrame, path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    timestamps = pd.to_datetime(chunk.iloc[:, 0], errors="coerce")
    if timestamps.isna().any():
        bad_rows = timestamps.index[timestamps.isna()].tolist()[:5]
        raise ValueError(f"Unparseable timestamps in {path}, rows {bad_rows}")
    values = chunk.iloc[:, 1:].to_numpy(dtype=np.float64)
    return pd.DatetimeIndex(timestamps), values


def _audit_source(path: Path) -> tuple[list[str], dict, pd.DataFrame]:
    headers = pd.read_csv(path, sep=";", nrows=0).columns.tolist()
    if len(headers) < 2:
        raise ValueError("The UCI file must contain a timestamp and client columns")
    client_ids = [str(column) for column in headers[1:]]
    n_clients = len(client_ids)
    quarter_count = np.zeros(n_clients, dtype=np.int64)
    missing_count = np.zeros(n_clients, dtype=np.int64)
    zero_count = np.zeros(n_clients, dtype=np.int64)
    nonpositive_count = np.zeros(n_clients, dtype=np.int64)
    eligible_missing_count = np.zeros(n_clients, dtype=np.int64)
    eligible_quarter_count = 0
    total_rows = 0
    previous_timestamp = None
    first_timestamp = None
    last_timestamp = None
    duplicate_timestamps = 0
    irregular_intervals = 0
    first_nonzero = np.full(n_clients, None, dtype=object)
    value_min = np.full(n_clients, np.inf)
    value_max = np.full(n_clients, -np.inf)

    for chunk in read_quarter_hour_chunks(path):
        timestamps, values = _parse_chunk(chunk, path)
        if values.shape[1] != n_clients:
            raise ValueError("Client columns changed while reading the source file")
        finite = np.isfinite(values)
        missing_count += (~finite).sum(axis=0)
        zero_count += (finite & (values == 0)).sum(axis=0)
        nonpositive_count += (finite & (values <= 0)).sum(axis=0)
        quarter_count += len(timestamps)
        if finite.any():
            value_min = np.minimum(value_min, np.where(finite, values, np.inf).min(axis=0))
            value_max = np.maximum(value_max, np.where(finite, values, -np.inf).max(axis=0))

        for col in np.flatnonzero(pd.isna(first_nonzero)):
            nonzero_rows = np.flatnonzero(finite[:, col] & (values[:, col] != 0))
            if len(nonzero_rows):
                first_nonzero[col] = timestamps[nonzero_rows[0]].isoformat(sep=" ")

        eligible_rows = (
            (timestamps >= FIRST_ELIGIBLE)
            & (timestamps < LAST_ELIGIBLE_EXCLUSIVE)
        )
        if eligible_rows.any():
            eligible_quarter_count += int(eligible_rows.sum())
            eligible_missing_count += (~finite[eligible_rows]).sum(axis=0)

        if len(timestamps):
            if first_timestamp is None:
                first_timestamp = timestamps[0]
            deltas = timestamps.to_series().diff().dropna()
            if previous_timestamp is not None:
                boundary_delta = timestamps[0] - previous_timestamp
                duplicate_timestamps += int(boundary_delta == ZERO_DELTA)
                irregular_intervals += int(
                    boundary_delta != QUARTER_DELTA
                )
            duplicate_timestamps += int(deltas.eq(ZERO_DELTA).sum())
            irregular_intervals += int(deltas.ne(QUARTER_DELTA).sum())
            previous_timestamp = timestamps[-1]
            last_timestamp = timestamps[-1]
        total_rows += len(chunk)

    expected_quarters = int(
        (LAST_ELIGIBLE_EXCLUSIVE - FIRST_ELIGIBLE).total_seconds() // 900
    )
    eligible_ids = [
        client_ids[index]
        for index in range(n_clients)
        if eligible_quarter_count == expected_quarters
        and eligible_missing_count[index] == 0
    ]
    client_audit = pd.DataFrame(
        {
            "unique_id": client_ids,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "first_nonzero_timestamp": first_nonzero,
            "quarter_hour_rows": quarter_count,
            "missing_values": missing_count,
            "zero_values": zero_count,
            "nonpositive_values": nonpositive_count,
            "eligible_quarter_hour_rows": eligible_quarter_count,
            "eligible_missing_values": eligible_missing_count,
            "min_kw": value_min,
            "max_kw": value_max,
            "complete_from_2012": [
                client_id in eligible_ids for client_id in client_ids
            ],
        }
    )
    audit = {
        "source_file": path.name,
        "source_sha256": _sha256(path),
        "source_size_bytes": path.stat().st_size,
        "source_rows": total_rows,
        "client_count": n_clients,
        "first_timestamp": first_timestamp.isoformat(sep=" ")
        if first_timestamp is not None
        else None,
        "last_timestamp": last_timestamp.isoformat(sep=" ")
        if last_timestamp is not None
        else None,
        "source_power_unit": "kW",
        "target_energy_unit": "kWh",
        "quarter_hour_energy_factor": 0.25,
        "timestamp_timezone": "Europe/Lisbon",
        "timestamp_representation": "local naive",
        "duplicate_timestamps": duplicate_timestamps,
        "irregular_15_minute_intervals": irregular_intervals,
        "missing_values": int(missing_count.sum()),
        "zero_values": int(zero_count.sum()),
        "nonpositive_values": int(nonpositive_count.sum()),
        "eligible_window_start": FIRST_ELIGIBLE.isoformat(sep=" "),
        "eligible_window_end_exclusive": LAST_ELIGIBLE_EXCLUSIVE.isoformat(sep=" "),
        "expected_quarter_hour_rows_per_client": expected_quarters,
        "eligible_quarter_hour_rows": eligible_quarter_count,
        "eligible_client_count": len(eligible_ids),
        "eligible_client_ids": eligible_ids,
        "timestamp_policy": (
            "Group quarter-hour kW readings by their local hour label and "
            "multiply each reading by 0.25 h. Incomplete boundary hours are "
            "excluded rather than imputed."
        ),
        "dst_policy": (
            "Preserve the UCI Portuguese local-wall-clock grid. UCI documents "
            "96 source intervals per day, with the spring gap represented by "
            "zeros and the autumn repeated hour aggregated in local time. The "
            "observed spring-hour nonzero counts are recorded below; source "
            "values are preserved rather than forced to zero."
        ),
    }
    return client_ids, audit, client_audit


def preprocess_dataset(
    input_path: Path,
    output_path: Path,
    audit_dir: Path,
    seed: int = SEED,
    max_clients: int | None = None,
) -> dict:
    """Audit the raw file and create an hourly long-format Parquet panel."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    client_ids, audit, client_audit = _audit_source(input_path)
    if audit["duplicate_timestamps"] or audit["irregular_15_minute_intervals"]:
        raise ValueError("Source time index contains duplicates or irregular intervals")
    if audit["missing_values"]:
        raise ValueError("Source file contains missing or non-finite consumption values")
    if audit["eligible_client_count"] != len(client_ids):
        raise ValueError(
            "Not all clients have complete coverage from 2012-01-01; "
            "inspect per_client_audit.csv before selecting a panel"
        )

    limit = max_clients if max_clients is not None else len(client_ids)
    selected_ids = choose_client_ids(
        audit["eligible_client_ids"], limit=limit, seed=seed
    )
    selected_indices = [client_ids.index(client_id) for client_id in selected_ids]
    audit["seed"] = seed
    audit["selected_client_count"] = len(selected_ids)
    audit["selected_client_ids"] = selected_ids
    audit["selected_hourly_rows"] = 0
    audit["hourly_rows_per_client"] = None

    temporary_path = output_path.with_name(output_path.name + ".part")
    temporary_path.unlink(missing_ok=True)
    writer = None
    previous_hour = None
    first_hour = None
    output_rows = 0
    eligible_hour_count = 0
    hours_by_day: dict[str, int] = {}
    spring_hour_nonzero_clients: dict[str, int] = {}
    pending_timestamps = pd.DatetimeIndex([])
    pending_values = np.empty((0, len(selected_indices)), dtype=np.float64)
    try:
        for chunk_index, chunk in enumerate(read_quarter_hour_chunks(input_path)):
            timestamps, all_values = _parse_chunk(chunk, input_path)
            timestamps = pending_timestamps.append(timestamps)
            values = np.vstack(
                [pending_values, all_values[:, selected_indices]]
            )
            (
                hourly_timestamps,
                hourly_values,
                pending_timestamps,
                pending_values,
            ) = _aggregate_available_hours(
                timestamps,
                values,
                allow_initial_partial=chunk_index == 0,
            )
            if len(hourly_timestamps) == 0:
                continue
            if previous_hour is not None:
                if hourly_timestamps[0] - previous_hour != HOUR_DELTA:
                    raise ValueError("Converted local-hour timestamps are not continuous")
            else:
                first_hour = hourly_timestamps[0]
            previous_hour = hourly_timestamps[-1]
            eligible_hour_count += int(
                (
                    (hourly_timestamps >= FIRST_ELIGIBLE)
                    & (hourly_timestamps < LAST_ELIGIBLE_EXCLUSIVE)
                ).sum()
            )
            day_counts = pd.Series(hourly_timestamps.date).value_counts()
            for day, count in day_counts.items():
                key = day.isoformat()
                hours_by_day[key] = hours_by_day.get(key, 0) + int(count)
            for day in SPRING_DST_DATES:
                dst_hour = pd.Timestamp(f"{day} 01:00")
                matches = np.flatnonzero(hourly_timestamps == dst_hour)
                if len(matches):
                    spring_hour_nonzero_clients[day] = int(
                        np.count_nonzero(hourly_values[matches[0]])
                    )
            frame = pd.DataFrame(
                {
                    "unique_id": np.repeat(selected_ids, len(hourly_timestamps)),
                    "ds": np.tile(hourly_timestamps.to_numpy(), len(selected_ids)),
                    "y": hourly_values.T.reshape(-1),
                }
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            output_rows += len(frame)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()
        if writer is None:
            raise ValueError("The source dataset contains no rows")

    if len(pending_timestamps) and (
        (pending_timestamps >= FIRST_ELIGIBLE)
        & (pending_timestamps < LAST_ELIGIBLE_EXCLUSIVE)
    ).any():
        temporary_path.unlink(missing_ok=True)
        raise ValueError("The source ends with an incomplete hour in the eligible period")
    eligible_days = {
        day: count
        for day, count in hours_by_day.items()
        if FIRST_ELIGIBLE.date().isoformat() <= day < LAST_ELIGIBLE_EXCLUSIVE.date().isoformat()
    }
    expected_days = (LAST_ELIGIBLE_EXCLUSIVE - FIRST_ELIGIBLE).days
    if len(eligible_days) != expected_days or any(
        count != 24 for count in eligible_days.values()
    ):
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            "The eligible period must contain 24 local hours for every day"
        )
    expected_hours = int(
        (LAST_ELIGIBLE_EXCLUSIVE - FIRST_ELIGIBLE).total_seconds() // 3600
    )
    panel_hours = int((previous_hour - first_hour) / HOUR_DELTA) + 1
    if output_rows != len(selected_ids) * panel_hours:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("Unexpected row count in the hourly panel")
    if eligible_hour_count != expected_hours:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("Converted panel does not cover every eligible local hour")

    temporary_path.replace(output_path)
    audit["selected_hourly_rows"] = output_rows
    audit["hourly_rows_per_client"] = output_rows // len(selected_ids)
    audit["eligible_hour_count"] = eligible_hour_count
    audit["eligible_days_with_24_hours"] = len(eligible_days)
    audit["hourly_first_timestamp"] = first_hour.isoformat(sep=" ")
    audit["hourly_last_timestamp"] = previous_hour.isoformat(sep=" ")
    audit["local_days_with_24_hours"] = sum(
        count == 24 for count in hours_by_day.values()
    )
    audit["incomplete_first_hour_dropped"] = (
        pd.Timestamp(audit["first_timestamp"])
        .floor("h")
        .isoformat(sep=" ")
        if pd.Timestamp(audit["first_timestamp"]).floor("h")
        != pd.Timestamp(audit["first_timestamp"])
        else None
    )
    audit["incomplete_trailing_hour_dropped"] = (
        pending_timestamps.floor("h")[0].isoformat(sep=" ")
        if len(pending_timestamps)
        else None
    )
    audit["spring_dst_hour_nonzero_client_counts"] = spring_hour_nonzero_clients
    audit["autumn_dst_day_hour_counts"] = {
        day: hours_by_day.get(day, 0) for day in AUTUMN_DST_DATES
    }
    audit["hourly_panel_path"] = str(output_path)

    selected_audit = client_audit[client_audit["unique_id"].isin(selected_ids)]
    selected_audit.to_csv(audit_dir / "per_client_audit.csv", index=False)
    pd.DataFrame({"unique_id": selected_ids}).to_csv(
        audit_dir / "client_ids.csv", index=False
    )
    audit_path = audit_dir / "dataset_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/LD2011_2014.txt"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/hourly.parquet")
    )
    parser.add_argument("--audit-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-clients", type=int)
    args = parser.parse_args()
    audit = preprocess_dataset(
        args.input,
        args.output,
        args.audit_dir,
        seed=args.seed,
        max_clients=args.max_clients,
    )
    summary = {
        key: audit[key]
        for key in (
            "source_rows",
            "client_count",
            "duplicate_timestamps",
            "irregular_15_minute_intervals",
            "missing_values",
            "zero_values",
            "eligible_client_count",
            "selected_client_count",
            "hourly_rows_per_client",
            "selected_hourly_rows",
            "eligible_hour_count",
            "hourly_panel_path",
        )
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
