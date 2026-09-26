"""Shared human-readable formatting for the CLI surfaces.

benchmark.py and demo.py both need to print the same quantities, so the
unit-scaling and table rules live here rather than being duplicated (and
drifting) between the two scripts.
"""

from __future__ import annotations

# Unit ladders: (suffix, multiplier applied to the base unit), smallest first.
_LADDERS = {
    "g": (("mg", 1000.0), ("g", 1.0), ("kg", 0.001)),
    "L": (("mL", 1000.0), ("L", 1.0)),
    "kWh": (("mWh", 1e6), ("Wh", 1000.0), ("kWh", 1.0)),
}


def pick_unit(values: list[float], unit: str) -> tuple[str, float]:
    """Choose one unit for a whole column, driven by its largest value.

    Formatting each cell independently makes a column flip between mWh and
    kWh mid-table, which is unreadable on a slide. The column instead commits
    to the largest unit that still keeps its peak value below 1000, so figures
    land in a readable 1-999 band rather than as 10,441 mWh.
    """
    ladder = _LADDERS[unit]
    peak = max((abs(v) for v in values), default=0.0)
    for suffix, multiplier in ladder:
        if peak * multiplier < 1000.0:
            return suffix, multiplier
    return ladder[-1]


def fmt_column(values: list[float], unit: str) -> list[str]:
    """Format a column of quantities in a single shared unit."""
    suffix, multiplier = pick_unit(values, unit)
    return [f"{v * multiplier:,.3f} {suffix}" for v in values]


def fmt(value: float, unit: str) -> str:
    """Format a single quantity, auto-scaling to a readable unit."""
    return fmt_column([value], unit)[0]


def table(headers: list[str], rows: list[list[str]], indent: str = "") -> str:
    """Render a fixed-width text table with a rule under the header."""
    if not rows:
        return indent + "  ".join(headers)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    head = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * w for w in widths)
    body = "\n".join(
        indent + "  ".join(r[i].ljust(widths[i]) for i in range(len(headers))).rstrip()
        for r in rows
    )
    return f"{indent}{head.rstrip()}\n{indent}{rule}\n{body}"
