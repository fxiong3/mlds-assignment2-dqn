"""CSV metric persistence and small analysis helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One row of training diagnostics."""

    episode: int
    reward: float
    length: int
    mean_max_q: float
    epsilon: float
    mean_loss: float
    mean_absolute_td_error: float
    action_counts: tuple[int, ...]


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    """Return a prefix-aware moving average with the same length as input."""

    if window <= 0:
        raise ValueError("window must be positive")
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return array
    cumulative = np.cumsum(array)
    result = np.empty_like(array)
    for index in range(array.size):
        start = max(0, index - window + 1)
        total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
        result[index] = total / (index - start + 1)
    return result


def write_episode_records(path: Path, records: Iterable[EpisodeRecord]) -> None:
    """Atomically replace the episode CSV with all current records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [field for field in EpisodeRecord.__dataclass_fields__]
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["action_counts"] = json.dumps(record.action_counts)
            writer.writerow(row)
    temporary_path.replace(path)


def read_episode_records(path: Path) -> list[EpisodeRecord]:
    """Read records written by :func:`write_episode_records`."""

    records: list[EpisodeRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                EpisodeRecord(
                    episode=int(row["episode"]),
                    reward=float(row["reward"]),
                    length=int(row["length"]),
                    mean_max_q=float(row["mean_max_q"]),
                    epsilon=float(row["epsilon"]),
                    mean_loss=float(row["mean_loss"]),
                    mean_absolute_td_error=float(row["mean_absolute_td_error"]),
                    action_counts=tuple(json.loads(row["action_counts"])),
                )
            )
    return records
