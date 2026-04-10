from __future__ import annotations


def format_bytes(value: int | float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.1f} {units[unit_index]}"


def format_speed(value: float) -> str:
    if value <= 0:
        return "0 B/s"
    return f"{format_bytes(value)}/s"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    remaining = max(0, int(seconds))
    hours, remaining = divmod(remaining, 3600)
    minutes, secs = divmod(remaining, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

