"""Small dependency-free terminal progress indicator."""


class ProgressBar:
    """Render progress at one-percent intervals on a single terminal line."""

    def __init__(self, label: str, total: int, width: int = 30) -> None:
        self._label = label
        self._total = max(total, 1)
        self._width = width
        self._last_percent = -1

    def update(self, completed: int) -> None:
        """Render when the integer percentage changes or work completes."""
        bounded = min(max(completed, 0), self._total)
        ratio = bounded / self._total
        percent = int(ratio * 100)
        if percent == self._last_percent and bounded != self._total:
            return
        self._last_percent = percent
        filled = min(int(ratio * self._width), self._width)
        bar = "#" * filled + "-" * (self._width - filled)
        ending = "\n" if bounded == self._total else ""
        print(
            f"\r  {self._label:<8} [{bar}] {ratio * 100:6.2f}%",
            end=ending,
            flush=True,
        )
