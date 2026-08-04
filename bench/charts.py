"""Chart output for the validation bench — ASCII for the terminal, SVG for docs.

Hand-rolled rather than matplotlib: the bench has to be runnable from a bare
checkout (`python bench/validate.py`), and adding a plotting dependency to prove
a caching claim would be a poor trade. The SVGs are plain text, so they diff
sensibly in git alongside the numbers they illustrate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

BLOCKS = "▏▎▍▌▋▊▉█"


# ------------------------------------------------------------------ ascii ---
def ascii_bars(rows: Sequence[tuple[str, float]], width: int = 44,
               fmt: Optional[Callable[[float], str]] = None,
               vmax: Optional[float] = None) -> str:
    """Horizontal bar chart with 1/8-cell resolution."""
    if not rows:
        return "(no data)"
    fmt = fmt or (lambda v: f"{v:,.0f}")
    top = vmax if vmax is not None else max(max(v for _, v in rows), 1e-9)
    label_w = max(len(str(l)) for l, _ in rows)
    val_w = max(len(fmt(v)) for _, v in rows)
    out = []
    for label, v in rows:
        cells = max(0.0, v) / top * width
        full = int(cells)
        rem = cells - full
        bar = "█" * full
        if rem > 1 / 16 and full < width:
            bar += BLOCKS[min(7, int(rem * 8))]
        out.append(f"{str(label):<{label_w}}  {bar:<{width}} {fmt(v):>{val_w}}")
    return "\n".join(out)


def ascii_series(series: dict[str, Sequence[float]], height: int = 12,
                 ymax: Optional[float] = None, ylabel_fmt: str = "{:>5.0%}",
                 xlabels: Optional[Sequence[str]] = None) -> str:
    """Overlaid line plot. Later series win a contested cell, so pass the
    series you most want to see last."""
    if not series:
        return "(no data)"
    n = max(len(v) for v in series.values())
    top = ymax if ymax is not None else max(
        (max(v) if v else 0) for v in series.values()) or 1.0
    marks = "·×○●▲■"
    grid = [[" "] * n for _ in range(height)]
    for si, (_, values) in enumerate(series.items()):
        for x, y in enumerate(values):
            row = height - 1 - int(round(min(max(y / top, 0.0), 1.0) * (height - 1)))
            grid[row][x] = marks[si % len(marks)]
    lines = []
    for r, row in enumerate(grid):
        frac = (height - 1 - r) / (height - 1)
        axis = ylabel_fmt.format(frac * top) if r % 2 == 0 else " " * 5
        lines.append(f"{axis} │{''.join(row)}")
    lines.append("      └" + "─" * n)
    if xlabels:
        lines.append("       " + "".join(xlabels)[:n])
    legend = "  ".join(f"{marks[i % len(marks)]} {name}"
                       for i, name in enumerate(series))
    lines.append("       " + legend)
    return "\n".join(lines)


# -------------------------------------------------------------------- svg ---
_PALETTE = ["#1f6f8b", "#e07a3f", "#4e9a51", "#a4485f", "#7a5ea8", "#8a8a8a"]
_FONT = "font-family='ui-monospace,Menlo,Consolas,monospace'"


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _frame(w: int, h: int, title: str, body: str, subtitle: str = "") -> str:
    sub = (f"<text x='{w/2}' y='44' text-anchor='middle' fill='#666' "
           f"font-size='12' {_FONT}>{_esc(subtitle)}</text>" if subtitle else "")
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
        f"viewBox='0 0 {w} {h}'>\n"
        f"<rect width='{w}' height='{h}' fill='#fdfdfc'/>\n"
        f"<text x='{w/2}' y='26' text-anchor='middle' fill='#111' "
        f"font-size='15' font-weight='600' {_FONT}>{_esc(title)}</text>\n"
        f"{sub}\n{body}\n</svg>\n")


def svg_lines(path: Path, title: str, series: dict[str, Sequence[float]],
              xlabel: str = "", ylabel: str = "", ymax: Optional[float] = None,
              ytick_fmt: Callable[[float], str] = lambda v: f"{v:.0%}",
              xtick_labels: Optional[Sequence[str]] = None,
              subtitle: str = "", markers: Iterable[tuple[int, str]] = (),
              width: int = 720, height: int = 360) -> Path:
    """Multi-series line chart. ``markers`` annotates x positions (e.g. prunes)."""
    L, R, T, B = 74, 150, 62, 52
    pw, ph = width - L - R, height - T - B
    n = max((len(v) for v in series.values()), default=1)
    top = ymax if ymax is not None else max(
        (max(v) if v else 0) for v in series.values()) or 1.0
    px = lambda i: L + (pw * i / max(n - 1, 1))
    py = lambda v: T + ph - (ph * min(max(v / top, 0.0), 1.0))

    b = [f"<rect x='{L}' y='{T}' width='{pw}' height='{ph}' fill='#fff' "
         f"stroke='#ddd'/>"]
    for k in range(5):
        v = top * k / 4
        y = py(v)
        b.append(f"<line x1='{L}' y1='{y:.1f}' x2='{L+pw}' y2='{y:.1f}' "
                 f"stroke='#eee'/>")
        b.append(f"<text x='{L-8}' y='{y+4:.1f}' text-anchor='end' fill='#666' "
                 f"font-size='11' {_FONT}>{_esc(ytick_fmt(v))}</text>")
    for i, xm in markers:
        b.append(f"<line x1='{px(i):.1f}' y1='{T}' x2='{px(i):.1f}' "
                 f"y2='{T+ph}' stroke='#c0392b' stroke-width='1' "
                 f"stroke-dasharray='4 3'/>")
        b.append(f"<text x='{px(i)+4:.1f}' y='{T+14}' fill='#c0392b' "
                 f"font-size='10' {_FONT}>{_esc(xm)}</text>")
    for si, (name, values) in enumerate(series.items()):
        color = _PALETTE[si % len(_PALETTE)]
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
        b.append(f"<polyline points='{pts}' fill='none' stroke='{color}' "
                 f"stroke-width='2'/>")
        for i, v in enumerate(values):
            b.append(f"<circle cx='{px(i):.1f}' cy='{py(v):.1f}' r='2.5' "
                     f"fill='{color}'/>")
        ly = T + 18 + si * 20
        b.append(f"<line x1='{L+pw+16}' y1='{ly}' x2='{L+pw+36}' y2='{ly}' "
                 f"stroke='{color}' stroke-width='2'/>")
        b.append(f"<text x='{L+pw+42}' y='{ly+4}' fill='#333' font-size='11' "
                 f"{_FONT}>{_esc(name)}</text>")
    if xtick_labels:
        step = max(1, n // 12)
        for i in range(0, n, step):
            if i < len(xtick_labels):
                b.append(f"<text x='{px(i):.1f}' y='{T+ph+16}' "
                         f"text-anchor='middle' fill='#666' font-size='10' "
                         f"{_FONT}>{_esc(xtick_labels[i])}</text>")
    if xlabel:
        b.append(f"<text x='{L+pw/2}' y='{height-12}' text-anchor='middle' "
                 f"fill='#333' font-size='12' {_FONT}>{_esc(xlabel)}</text>")
    if ylabel:
        b.append(f"<text x='16' y='{T+ph/2}' fill='#333' font-size='12' "
                 f"transform='rotate(-90 16 {T+ph/2})' text-anchor='middle' "
                 f"{_FONT}>{_esc(ylabel)}</text>")
    path.write_text(_frame(width, height, title, "\n".join(b), subtitle),
                    encoding="utf-8")
    return path


def svg_bars(path: Path, title: str, rows: Sequence[tuple[str, float]],
             value_fmt: Callable[[float], str] = lambda v: f"{v:,.0f}",
             subtitle: str = "", ylabel: str = "",
             highlight: Optional[str] = None,
             width: int = 720, height: int = 330) -> Path:
    """Vertical bars, one per row. ``highlight`` colours one label differently."""
    L, R, T, B = 80, 28, 62, 62
    pw, ph = width - L - R, height - T - B
    top = max((v for _, v in rows), default=1.0) or 1.0
    slot = pw / max(len(rows), 1)
    bw = slot * 0.62
    b = [f"<line x1='{L}' y1='{T+ph}' x2='{L+pw}' y2='{T+ph}' stroke='#bbb'/>"]
    for k in range(5):
        v = top * k / 4
        y = T + ph - ph * v / top
        b.append(f"<line x1='{L}' y1='{y:.1f}' x2='{L+pw}' y2='{y:.1f}' "
                 f"stroke='#eee'/>")
        b.append(f"<text x='{L-8}' y='{y+4:.1f}' text-anchor='end' fill='#666' "
                 f"font-size='11' {_FONT}>{_esc(value_fmt(v))}</text>")
    for i, (label, v) in enumerate(rows):
        x = L + slot * i + (slot - bw) / 2
        h = ph * max(v, 0) / top
        color = (_PALETTE[3] if highlight and label == highlight
                 else _PALETTE[i % len(_PALETTE)])
        b.append(f"<rect x='{x:.1f}' y='{T+ph-h:.1f}' width='{bw:.1f}' "
                 f"height='{h:.1f}' fill='{color}' rx='2'/>")
        b.append(f"<text x='{x+bw/2:.1f}' y='{T+ph-h-6:.1f}' "
                 f"text-anchor='middle' fill='#333' font-size='11' "
                 f"{_FONT}>{_esc(value_fmt(v))}</text>")
        for j, part in enumerate(str(label).split("\n")):
            b.append(f"<text x='{x+bw/2:.1f}' y='{T+ph+16+j*13:.1f}' "
                     f"text-anchor='middle' fill='#444' font-size='10' "
                     f"{_FONT}>{_esc(part)}</text>")
    if ylabel:
        b.append(f"<text x='16' y='{T+ph/2}' fill='#333' font-size='12' "
                 f"transform='rotate(-90 16 {T+ph/2})' text-anchor='middle' "
                 f"{_FONT}>{_esc(ylabel)}</text>")
    path.write_text(_frame(width, height, title, "\n".join(b), subtitle),
                    encoding="utf-8")
    return path


def svg_stacked(path: Path, title: str, labels: Sequence[str],
                parts: Sequence[tuple[str, Sequence[float]]],
                value_fmt: Callable[[float], str] = lambda v: f"{v:,.0f}",
                subtitle: str = "", ylabel: str = "",
                width: int = 720, height: int = 360) -> Path:
    """Stacked bars — used for cached vs freshly-billed prompt tokens."""
    L, R, T, B = 84, 150, 62, 56
    pw, ph = width - L - R, height - T - B
    totals = [sum(p[1][i] for p in parts) for i in range(len(labels))]
    top = max(totals, default=1.0) or 1.0
    slot = pw / max(len(labels), 1)
    bw = slot * 0.58
    b = [f"<line x1='{L}' y1='{T+ph}' x2='{L+pw}' y2='{T+ph}' stroke='#bbb'/>"]
    for k in range(5):
        v = top * k / 4
        y = T + ph - ph * v / top
        b.append(f"<line x1='{L}' y1='{y:.1f}' x2='{L+pw}' y2='{y:.1f}' "
                 f"stroke='#eee'/>")
        b.append(f"<text x='{L-8}' y='{y+4:.1f}' text-anchor='end' fill='#666' "
                 f"font-size='11' {_FONT}>{_esc(value_fmt(v))}</text>")
    for i, label in enumerate(labels):
        x = L + slot * i + (slot - bw) / 2
        acc = 0.0
        for pi, (_, values) in enumerate(parts):
            v = values[i]
            h = ph * v / top
            b.append(f"<rect x='{x:.1f}' y='{T+ph-h-acc:.1f}' width='{bw:.1f}' "
                     f"height='{h:.1f}' fill='{_PALETTE[pi % len(_PALETTE)]}' "
                     f"rx='1'/>")
            acc += h
        b.append(f"<text x='{x+bw/2:.1f}' y='{T+ph-acc-6:.1f}' "
                 f"text-anchor='middle' fill='#333' font-size='10' "
                 f"{_FONT}>{_esc(value_fmt(totals[i]))}</text>")
        for j, part in enumerate(str(label).split("\n")):
            b.append(f"<text x='{x+bw/2:.1f}' y='{T+ph+16+j*13:.1f}' "
                     f"text-anchor='middle' fill='#444' font-size='10' "
                     f"{_FONT}>{_esc(part)}</text>")
    for pi, (name, _) in enumerate(parts):
        ly = T + 18 + pi * 20
        b.append(f"<rect x='{L+pw+16}' y='{ly-8}' width='14' height='11' "
                 f"fill='{_PALETTE[pi % len(_PALETTE)]}' rx='2'/>")
        b.append(f"<text x='{L+pw+36}' y='{ly+2}' fill='#333' font-size='11' "
                 f"{_FONT}>{_esc(name)}</text>")
    if ylabel:
        b.append(f"<text x='16' y='{T+ph/2}' fill='#333' font-size='12' "
                 f"transform='rotate(-90 16 {T+ph/2})' text-anchor='middle' "
                 f"{_FONT}>{_esc(ylabel)}</text>")
    path.write_text(_frame(width, height, title, "\n".join(b), subtitle),
                    encoding="utf-8")
    return path
