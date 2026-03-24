# datacore/progress.py
"""
ProgressReporter - Emit standardised progress lines for the LM Data Tools webapp.

The webapp monitors subprocess stdout and parses lines that match::

    PROGRESS {current}/{total}

Using a **single** regex ``r"PROGRESS (\\d+)/(\\d+)"`` — no per-tool patterns
needed.

Usage
-----
    from datacore.progress import ProgressReporter

    reporter = ProgressReporter(total=len(items), phase="Generating answers")
    for i, item in enumerate(items):
        # … do work …
        reporter.update(i + 1)

    reporter.done()   # emits PROGRESS total/total (optional, guarantees 100%)
"""


class ProgressReporter:
    """
    Emits ``PROGRESS {current}/{total} [{phase}]`` lines to stdout.

    Parameters
    ----------
    total:
        Total number of items to process.
    phase:
        Human-readable label appended after the counts (e.g. "Generating").
        Optional — omit for a bare progress line.
    silent:
        If ``True`` suppress all output (useful in unit tests).
    """

    def __init__(self, total: int, phase: str = "", silent: bool = False):
        self.total = max(1, total)  # guard against zero-division
        self.phase = phase
        self.silent = silent
        self._last = 0

    def update(self, current: int, phase: str = None) -> None:
        """
        Emit a ``PROGRESS`` line for *current* (1-based) out of *total*.

        Parameters
        ----------
        current:
            The 1-based index of the item just completed.
        phase:
            Overrides ``self.phase`` for this single call only.
        """
        if self.silent:
            return
        label = phase if phase is not None else self.phase
        parts = [f"PROGRESS {current}/{self.total}"]
        if label:
            parts.append(label)
        print(" ".join(parts), flush=True)
        self._last = current

    def done(self) -> None:
        """Emit a final ``PROGRESS total/total`` line (guarantees 100%)."""
        self.update(self.total)
