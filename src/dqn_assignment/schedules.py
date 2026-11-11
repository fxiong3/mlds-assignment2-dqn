"""Small, deterministic hyperparameter schedules."""


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    """Interpolate from start to end and clamp outside the schedule interval."""

    if duration <= 0:
        raise ValueError("duration must be positive")
    progress = min(max(step, 0) / duration, 1.0)
    return start + progress * (end - start)
