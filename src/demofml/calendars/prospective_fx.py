"""Frozen five-minute New York FX calendar for Campaign 2."""

from datetime import UTC, datetime, time, timedelta
from functools import lru_cache
from importlib import metadata, resources
from zoneinfo import ZoneInfo

CALENDAR_ID = "prospective-fx-5m-v1"
TIMEZONE_NAME = "America/New_York"
TZDATA_VERSION = "2025.2"
INTERVAL = timedelta(minutes=5)
_SUNDAY_OPEN = time(17, 5)
_FRIDAY_STOP = time(16, 0)


@lru_cache(maxsize=1)
def _new_york() -> ZoneInfo:
    installed = metadata.version("tzdata")
    if installed != TZDATA_VERSION:
        raise RuntimeError(
            f"{CALENDAR_ID} requires tzdata {TZDATA_VERSION}, found {installed}"
        )
    zone = resources.files("tzdata.zoneinfo").joinpath("America", "New_York")
    with zone.open("rb") as handle:
        return ZoneInfo.from_file(handle, key=TIMEZONE_NAME)


def _validate_boundary(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.second or value.microsecond or value.minute % 5:
        raise ValueError(f"{name} must align to the five-minute UTC grid")


def is_expected_decision_boundary(boundary: datetime) -> bool:
    """Return whether a UTC boundary belongs to the fixed weekly session."""
    _validate_boundary(boundary, "boundary")
    local = boundary.astimezone(_new_york())
    weekday = local.weekday()
    local_time = local.timetz().replace(tzinfo=None)
    if weekday == 6:
        return local_time >= _SUNDAY_OPEN
    if weekday <= 3:
        return True
    if weekday == 4:
        return local_time < _FRIDAY_STOP
    return False


def expected_decision_boundaries(
    start: datetime,
    end: datetime,
) -> tuple[datetime, ...]:
    """Generate expected boundaries in half-open UTC interval ``[start, end)``."""
    _validate_boundary(start, "start")
    _validate_boundary(end, "end")
    if end < start:
        raise ValueError("end must not precede start")
    zone = _new_york()
    boundaries: list[datetime] = []
    current = start.astimezone(UTC)
    while current < end:
        local = current.astimezone(zone)
        weekday = local.weekday()
        local_time = local.timetz().replace(tzinfo=None)
        if (
            (weekday == 6 and local_time >= _SUNDAY_OPEN)
            or weekday <= 3
            or (weekday == 4 and local_time < _FRIDAY_STOP)
        ):
            boundaries.append(current)
        current += INTERVAL
    return tuple(boundaries)
