"""Layout-aware table extraction for invoices, statements, ledgers and reports.

The design goal is that *one* reconstruction algorithm serves every kind of
input. Each page is reduced to positioned text :class:`Fragment` objects --
either from the PDF text layer or from OCR on a rendered image -- and from
there the same code recovers rows, columns and cells.

Per page the engine picks a strategy:

===================  =========================================================
ruled + digital      pdfplumber's line-based table finder (exact cell grid)
borderless digital   whitespace-projection column detection over PDF words
ruled + scanned      OpenCV morphology finds the grid, OCR fills the cells
borderless scanned   whitespace-projection column detection over OCR boxes
===================  =========================================================

Tables that continue across pages are stitched back together afterwards and
repeated headers are dropped, so a 40-page statement yields one clean sheet.
"""

from __future__ import annotations

import io
import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ocr_engine import Fragment, OCREngine, binarize, deskew

logger = logging.getLogger(__name__)

# --- tuning knobs ----------------------------------------------------------

OCR_DPI = 300                 # render resolution for scanned pages
MIN_WORDS_DIGITAL = 12        # fewer words than this => treat page as scanned
MIN_TABLE_ROWS = 2            # a block needs at least this many rows to count
MIN_COLUMN_FILL = 0.15        # a column must be non-empty in >=15% of rows
MAX_ROW_HEIGHT_FACTOR = 2.2   # guards row clustering against tall glyphs
HEADER_SIMILARITY = 0.6       # row/header token overlap that means "repeat"


@dataclass
class Table:
    """One reconstructed table, still as raw strings."""

    rows: List[List[str]]
    page_numbers: List[int] = field(default_factory=list)
    source: str = "text"
    title: Optional[str] = None
    column_bounds: Optional[List[float]] = None
    has_header: bool = True

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def header(self) -> List[str]:
        return self.rows[0] if (self.has_header and self.rows) else []

    @property
    def body(self) -> List[List[str]]:
        return self.rows[1:] if (self.has_header and self.rows) else self.rows


# ---------------------------------------------------------------------------
# Row clustering
# ---------------------------------------------------------------------------


def cluster_rows(fragments: Sequence[Fragment]) -> List[List[Fragment]]:
    """Group fragments into visual lines by vertical overlap.

    A running band is grown fragment by fragment, but it is not allowed to grow
    past ``MAX_ROW_HEIGHT_FACTOR`` times the median glyph height -- otherwise a
    single oversized heading would swallow the rows beneath it.
    """
    if not fragments:
        return []

    frags = sorted(fragments, key=lambda f: (f.top, f.x0))
    heights = [f.height for f in frags if f.height > 0]
    med_h = statistics.median(heights) if heights else 8.0
    max_band = MAX_ROW_HEIGHT_FACTOR * med_h

    rows: List[List[Fragment]] = []
    current = [frags[0]]
    top, bottom = frags[0].top, frags[0].bottom

    for frag in frags[1:]:
        overlap = min(bottom, frag.bottom) - max(top, frag.top)
        centre_close = abs(frag.cy - (top + bottom) / 2) < 0.45 * med_h
        fits = (overlap > 0.35 * min(med_h, max(frag.height, 1.0))) or centre_close

        if fits and (max(bottom, frag.bottom) - min(top, frag.top)) <= max_band:
            current.append(frag)
            top, bottom = min(top, frag.top), max(bottom, frag.bottom)
        else:
            rows.append(sorted(current, key=lambda f: f.x0))
            current = [frag]
            top, bottom = frag.top, frag.bottom

    rows.append(sorted(current, key=lambda f: f.x0))
    return rows


def _median_height(rows: Sequence[Sequence[Fragment]]) -> float:
    heights = [f.height for row in rows for f in row if f.height > 0]
    return statistics.median(heights) if heights else 8.0


# ---------------------------------------------------------------------------
# Column detection (whitespace projection)
# ---------------------------------------------------------------------------


def _whitespace_separators(
    rows: Sequence[Sequence[Fragment]], min_gap: float, bin_size: float = 0.5
) -> List[float]:
    """Find x positions where a vertical whitespace corridor runs down the block.

    Occupancy is projected onto the x axis; a run of unoccupied bins at least
    ``min_gap`` wide is a column separator. A small tolerance lets a stray
    descender or a merged cell in one row out of many cross the corridor
    without destroying it.
    """
    xs = [f.x0 for row in rows for f in row] + [f.x1 for row in rows for f in row]
    if not xs:
        return []

    left, right = min(xs), max(xs)
    span = right - left
    if span <= 0:
        return []

    n_bins = max(1, int(np.ceil(span / bin_size)))
    occupancy = np.zeros(n_bins, dtype=np.int32)

    for row in rows:
        # Mark once per row so a wide row cannot dominate the histogram.
        row_mask = np.zeros(n_bins, dtype=bool)
        for frag in row:
            start = int((frag.x0 - left) / bin_size)
            end = int(np.ceil((frag.x1 - left) / bin_size))
            row_mask[max(0, start):min(n_bins, max(end, start + 1))] = True
        occupancy += row_mask

    tolerance = max(0, int(0.02 * len(rows)))
    free = occupancy <= tolerance

    separators: List[float] = []
    run_start: Optional[int] = None
    for i, is_free in enumerate(free):
        if is_free and run_start is None:
            run_start = i
        elif not is_free and run_start is not None:
            _emit_separator(separators, run_start, i, left, bin_size, min_gap)
            run_start = None
    # A trailing run touches the right edge of the block, so it is not interior.

    return separators


def _emit_separator(
    out: List[float], start: int, end: int, left: float, bin_size: float, min_gap: float
) -> None:
    if start == 0:
        return  # leading margin, not an interior separator
    if (end - start) * bin_size >= min_gap:
        out.append(left + (start + end) / 2.0 * bin_size)


def _alignment_score(rows: Sequence[Sequence[Fragment]], separators: Sequence[float]) -> float:
    """Rate a candidate column split by how consistently its cells align.

    Real columns are either left- or right-aligned, so one of those edges has a
    small spread. A split that accidentally cut through flowing prose has both
    edges scattered and scores badly.
    """
    bounds = [-1e9, *separators, 1e9]
    n_cols = len(bounds) - 1
    if n_cols < 2:
        return 0.0

    edges: List[Tuple[List[float], List[float]]] = [([], []) for _ in range(n_cols)]
    filled = [0] * n_cols

    for row in rows:
        seen = set()
        for frag in row:
            idx = _column_of(frag, bounds)
            lefts, rights = edges[idx]
            lefts.append(frag.x0)
            rights.append(frag.x1)
            seen.add(idx)
        for idx in seen:
            filled[idx] += 1

    n_rows = max(1, len(rows))
    scores = []
    for idx in range(n_cols):
        lefts, rights = edges[idx]
        if len(lefts) < 2:
            scores.append(0.5)  # too little evidence either way
            continue
        width = max(max(rights) - min(lefts), 1.0)
        spread = min(statistics.pstdev(lefts), statistics.pstdev(rights))
        scores.append(max(0.0, 1.0 - spread / width))

    alignment = sum(scores) / n_cols
    coverage = sum(min(1.0, f / n_rows / MIN_COLUMN_FILL) for f in filled) / n_cols
    return alignment * min(1.0, coverage)


def _column_of(frag: Fragment, bounds: Sequence[float]) -> int:
    """Column index whose span overlaps the fragment most."""
    best_idx, best_overlap = 0, -1.0
    for i in range(len(bounds) - 1):
        overlap = min(frag.x1, bounds[i + 1]) - max(frag.x0, bounds[i])
        if overlap > best_overlap:
            best_idx, best_overlap = i, overlap
    return best_idx


def detect_columns(rows: Sequence[Sequence[Fragment]]) -> List[float]:
    """Choose the column separators that best explain the block.

    Several gap widths are tried from coarse to fine and scored; the finest
    split that still looks like genuine columns wins.
    """
    med_h = _median_height(rows)
    candidates = [1.6, 1.2, 0.9, 0.7, 0.55, 0.45]

    best_seps: List[float] = []
    best_score = 0.0
    for factor in candidates:
        seps = _whitespace_separators(rows, min_gap=max(2.5, factor * med_h))
        if not seps:
            continue
        score = _alignment_score(rows, seps)
        # Prefer a finer split only when it is at least as convincing.
        if score > best_score + 0.02 or (not best_seps and score > 0):
            best_seps, best_score = seps, score

    return best_seps


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------


def _cells_from_rows(
    rows: Sequence[Sequence[Fragment]], separators: Sequence[float]
) -> List[List[str]]:
    bounds = [-1e9, *separators, 1e9]
    n_cols = len(bounds) - 1

    grid: List[List[str]] = []
    for row in rows:
        cells: List[List[Fragment]] = [[] for _ in range(n_cols)]
        for frag in row:
            cells[_column_of(frag, bounds)].append(frag)
        grid.append([
            " ".join(f.text for f in sorted(bucket, key=lambda f: f.x0)).strip()
            for bucket in cells
        ])
    return grid


def _drop_empty_columns(grid: List[List[str]], bounds: Optional[List[float]]) -> Tuple[List[List[str]], Optional[List[float]]]:
    if not grid:
        return grid, bounds
    n_cols = max(len(r) for r in grid)
    keep = [
        i for i in range(n_cols)
        if any(i < len(row) and row[i].strip() for row in grid)
    ]
    if len(keep) == n_cols:
        return grid, bounds
    trimmed = [[row[i] if i < len(row) else "" for i in keep] for row in grid]
    new_bounds = [bounds[i] for i in keep if bounds and i < len(bounds)] if bounds else None
    return trimmed, new_bounds


def _merge_alignment_splits(
    grid: List[List[str]], bounds: Optional[List[float]]
) -> Tuple[List[List[str]], Optional[List[float]]]:
    """Rejoin a column that alignment split down the middle.

    Invoice templates left-align a heading like ``Unit price`` but right-align
    the figures under it. When the two never overlap horizontally, a clear
    whitespace corridor runs the height of the column and the projection reads
    it as a column boundary -- turning six columns into ten, with every heading
    one column to the right of its own data.

    The signature is unmistakable: of the two candidates, one holds nothing but
    the heading and the other nothing but the body. Columns that are merely
    mutually exclusive -- ``Withdrawal`` and ``Deposit`` on a statement -- both
    carry a heading *and* body values, so they are left alone.
    """
    if len(grid) < 3:
        return grid, bounds

    while True:
        n_cols = max(len(r) for r in grid)
        pair = next(
            (i for i in range(n_cols - 1) if _is_alignment_split(grid, i, i + 1)), None
        )
        if pair is None:
            return grid, bounds

        grid = [
            row[:pair] + [" ".join(p for p in (row[pair], row[pair + 1]) if p.strip()).strip()]
            + row[pair + 2:]
            for row in (r + [""] * (n_cols - len(r)) for r in grid)
        ]
        if bounds and pair < len(bounds):
            bounds = bounds[:pair] + bounds[pair + 1:]


def _is_alignment_split(grid: Sequence[Sequence[str]], a: int, b: int) -> bool:
    def get(row: Sequence[str], i: int) -> str:
        return row[i].strip() if i < len(row) else ""

    header, body = grid[0], grid[1:]
    a_head, b_head = bool(get(header, a)), bool(get(header, b))
    a_body = any(get(row, a) for row in body)
    b_body = any(get(row, b) for row in body)

    return (a_head and not a_body and b_body and not b_head) or (
        b_head and not b_body and a_body and not a_head
    )


def _merge_wrapped_rows(grid: List[List[str]]) -> List[List[str]]:
    """Fold continuation lines back into the row they belong to.

    An invoice line item whose description wraps produces a follow-up row where
    every column except the text one is blank. Those get appended to the
    previous row rather than becoming rows of their own.
    """
    if len(grid) < 2:
        return grid

    n_cols = max(len(r) for r in grid)
    if n_cols < 3:
        return grid

    merged: List[List[str]] = [list(grid[0])]
    for row in grid[1:]:
        filled = [i for i, cell in enumerate(row) if cell.strip()]
        prev_filled = sum(1 for cell in merged[-1] if cell.strip())
        is_continuation = (
            len(filled) == 1
            and prev_filled >= 3
            and not _looks_numeric(row[filled[0]])
        )
        if is_continuation:
            idx = filled[0]
            while len(merged[-1]) <= idx:
                merged[-1].append("")
            merged[-1][idx] = f"{merged[-1][idx]} {row[idx]}".strip()
        else:
            merged.append(list(row))
    return merged


_NUMERIC_RE = re.compile(r"^[\s(\[]*[-+]?[₹$€£¥]?\s*[\d,]+(?:\.\d+)?\s*%?[\s)\]]*(?:CR|DR|Cr|Dr)?$")


def _looks_numeric(text: str) -> bool:
    return bool(_NUMERIC_RE.match(text.strip())) if text.strip() else False


# ---------------------------------------------------------------------------
# Block segmentation
# ---------------------------------------------------------------------------


def _row_is_tabular(row: Sequence[Fragment], min_gap: float) -> bool:
    """True when the line has at least two groups separated by a wide gap."""
    if len(row) < 2:
        return False
    ordered = sorted(row, key=lambda f: f.x0)
    return any(
        ordered[i + 1].x0 - ordered[i].x1 >= min_gap
        for i in range(len(ordered) - 1)
    )


def segment_blocks(
    rows: Sequence[Sequence[Fragment]],
) -> List[Tuple[List[List[Fragment]], Optional[str]]]:
    """Split a page's lines into table blocks, each with an optional title.

    Consecutive tabular lines form a block. A single non-tabular line inside a
    block is kept (it is usually a wrapped description or a sub-total caption);
    two in a row end it. The last prose line above a block becomes its title.
    """
    med_h = _median_height(rows)
    min_gap = max(3.0, 0.9 * med_h)
    flags = [_row_is_tabular(row, min_gap) for row in rows]

    blocks: List[Tuple[List[List[Fragment]], Optional[str]]] = []
    i = 0
    while i < len(rows):
        if not flags[i]:
            i += 1
            continue

        start = i
        gap_run = 0
        j = i
        while j < len(rows):
            if flags[j]:
                gap_run = 0
                j += 1
            elif gap_run == 0 and j + 1 < len(rows) and flags[j + 1]:
                gap_run = 1
                j += 1
            else:
                break
        end = j

        for offset, chunk in _split_on_vertical_gaps(rows[start:end]):
            if len(chunk) >= MIN_TABLE_ROWS:
                blocks.append((chunk, _title_above(rows, flags, start + offset)))
        i = max(end, i + 1)

    return blocks


def _split_on_vertical_gaps(
    block: Sequence[Sequence[Fragment]],
) -> List[Tuple[int, List[List[Fragment]]]]:
    """Cut a run of tabular lines wherever the vertical rhythm breaks.

    Two unrelated tables stacked on one page -- an address grid above a list of
    line items -- are both "tabular" and would otherwise be analysed as a
    single block, giving column boundaries that fit neither. The extra
    whitespace between them is the tell: rows inside one table keep a steady
    pitch. Returns each chunk with its offset into ``block``.
    """
    if len(block) < 4:
        return [(0, [list(r) for r in block])]

    tops = [min(f.top for f in row) for row in block]
    gaps = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    pitch = statistics.median(gaps)
    limit = max(1.7 * pitch, pitch + 6.0)

    chunks: List[Tuple[int, List[List[Fragment]]]] = []
    start = 0
    current = [list(block[0])]
    for i, gap in enumerate(gaps):
        if gap > limit:
            chunks.append((start, current))
            start, current = i + 1, []
        current.append(list(block[i + 1]))
    chunks.append((start, current))
    return chunks


def _title_above(
    rows: Sequence[Sequence[Fragment]], flags: Sequence[bool], start: int
) -> Optional[str]:
    """Name the table after the nearest heading above it.

    ``Label: value`` lines are skipped -- on a statement the lines directly
    above the table are account details, and the real heading (the bank name)
    sits above those.
    """
    for k in range(start - 1, max(-1, start - 7), -1):
        if flags[k]:
            break
        text = " ".join(f.text for f in sorted(rows[k], key=lambda f: f.x0)).strip()
        if not (2 < len(text) <= 80) or _KV_RE.match(text):
            continue
        return text
    return None


def _sparse_columns(grid: Sequence[Sequence[str]], threshold: float) -> bool:
    """True when some column is blank in most rows.

    Page furniture -- a footer sentence with a lone ``Page 3 of 9`` off to the
    right -- lines up like a two-column table but leaves one column mostly
    empty. A real table fills both.
    """
    if not grid:
        return True
    n_cols = max(len(r) for r in grid)
    for col in range(n_cols):
        filled = sum(1 for row in grid if col < len(row) and row[col].strip())
        if filled / len(grid) < threshold:
            return True
    return False


def _looks_like_prose(grid: Sequence[Sequence[str]]) -> bool:
    """Reject narrow blocks whose cells are really sentences."""
    if not grid or max(len(r) for r in grid) > 2:
        return False
    cells = [c.strip() for row in grid for c in row if c.strip()]
    if not cells:
        return True
    wordy = sum(1 for c in cells if len(c) > 35 and c.count(" ") >= 5)
    return wordy / len(cells) >= 0.35


def _column_evidence(block: Sequence[Sequence[Fragment]]) -> List[List[Fragment]]:
    """Choose which rows get to define the column boundaries.

    Headings are the least reliable witnesses in a table: a template will
    left-align ``Unit price`` over figures that are right-aligned beneath it,
    leaving a clear vertical corridor *inside* a single column. Letting the
    body rows vote instead keeps that corridor from being read as a boundary,
    and the heading is mapped onto the resulting columns afterwards.
    """
    body = [list(row) for row in block[1:]]
    return body if len(body) >= 2 else [list(row) for row in block]


def _is_key_value_block(grid: Sequence[Sequence[str]]) -> bool:
    """Detect an address/metadata block masquerading as a narrow table.

    Invoice headers ("Invoice No: ...", "GSTIN: ...") lay out in columns and
    would otherwise become a two-column table. Their content is more useful on
    the Document Details sheet, so they are rejected here.
    """
    if not grid or max(len(r) for r in grid) > 3:
        return False
    cells = [c.strip() for row in grid for c in row if c.strip()]
    if not cells:
        return True
    return sum(1 for c in cells if _KV_RE.match(c)) / len(cells) >= 0.5


def _fragments_to_tables(
    fragments: Sequence[Fragment], page_number: int, source: str
) -> List[Table]:
    rows = cluster_rows(fragments)
    tables: List[Table] = []

    for block, title in segment_blocks(rows):
        separators = detect_columns(_column_evidence(block))
        if not separators:
            continue
        grid = _cells_from_rows(block, separators)
        grid, separators = _drop_empty_columns(grid, list(separators))
        grid, separators = _merge_alignment_splits(grid, separators)
        grid = _merge_wrapped_rows(grid)
        grid = [row for row in grid if any(cell.strip() for cell in row)]
        n_cols = max((len(r) for r in grid), default=0)
        if len(grid) < MIN_TABLE_ROWS or n_cols < 2:
            continue
        if n_cols == 2 and (len(grid) < 3 or _sparse_columns(grid, 0.6)):
            continue  # too little structure to call it a table
        if _is_key_value_block(grid) or _looks_like_prose(grid):
            continue
        tables.append(
            Table(
                rows=grid,
                page_numbers=[page_number],
                source=source,
                title=title,
                column_bounds=list(separators) if separators else None,
            )
        )
    return tables


# ---------------------------------------------------------------------------
# Digital pages
# ---------------------------------------------------------------------------


def _page_words(page) -> List[Fragment]:
    words = page.extract_words(
        keep_blank_chars=False,
        use_text_flow=False,
        extra_attrs=["upright"],
    )
    return [
        Fragment(w["text"], float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))
        for w in words
        if w.get("upright", True) and w["text"].strip()
    ]


def _has_ruling_lines(page) -> bool:
    """Is it worth asking pdfplumber to read cells off the page's rules?

    Kept deliberately loose. Templates draw their borders in all sorts of ways
    -- strokes, hairline rectangles, filled bands -- and when real rules exist
    they give exact cell boundaries, which beats anything inferred from
    whitespace. Tables the lattice pass fails to find still fall through to the
    text path, so a false positive here costs nothing.
    """
    horizontal = [e for e in page.edges if e["orientation"] == "h" and e["x1"] - e["x0"] > 30]
    vertical = [e for e in page.edges if e["orientation"] == "v" and e["bottom"] - e["top"] > 8]
    return len(horizontal) >= 2 and len(vertical) >= 1


LATTICE_PROBE = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 4,
    "join_tolerance": 4,
    "intersection_tolerance": 4,
}


def _adaptive_tolerance(positions: Sequence[float]) -> float:
    """How far apart two rules must be before they count as different columns.

    Many exporters draw a single visible border as a stack of three or four
    hairline rectangles a few points apart. Those stacks have to collapse or a
    six-column table is read as twenty. The gap distribution gives it away:
    within a stack the gaps are tiny and uniform, between real columns they
    jump by an order of magnitude, so the tolerance is placed at the widest
    relative jump in the sorted gaps.

    Returns a small default when there is no such jump -- borders drawn as
    single strokes need no collapsing, and over-snapping them would merge
    genuinely narrow columns.
    """
    ordered = sorted(positions)
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > 0.1]
    if len(gaps) < 3:
        return 3.0

    gaps.sort()
    best_ratio, split = 0.0, None
    for i in range(len(gaps) - 1):
        if gaps[i] > 14.0:  # already wider than any border stack
            break
        ratio = gaps[i + 1] / max(gaps[i], 0.5)
        if ratio > best_ratio:
            best_ratio, split = ratio, gaps[i]

    if split is None or best_ratio < 2.0:
        return 3.0
    return min(12.0, max(2.0, split + 1.0))


def _cluster_positions(positions: Sequence[float], tolerance: float) -> List[List[float]]:
    ordered = sorted(positions)
    clusters: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def _representative(cluster: Sequence[float], spans: Sequence[Tuple[float, float]]) -> float:
    """Pick the grid line that does not cut through a word.

    Collapsing a stack of borders to its mean can land the boundary in the
    middle of the text beside it -- that is what turns ``Ship to`` into ``S``
    and ``hip to``. Any member of the cluster is an equally valid boundary, so
    prefer one that crosses no word box.
    """
    candidates = [cluster[0], cluster[-1], sum(cluster) / len(cluster)]
    for candidate in candidates:
        if not any(lo < candidate < hi for lo, hi in spans):
            return candidate
    return sum(cluster) / len(cluster)


def _grid_lines(edges: Sequence[dict], axis: str, spans: Sequence[Tuple[float, float]]) -> List[float]:
    if axis == "v":
        raw = [e["x0"] for e in edges if e["orientation"] == "v"]
        raw += [e["x1"] for e in edges if e["orientation"] == "v"]
    else:
        raw = [e["top"] for e in edges if e["orientation"] == "h"]
        raw += [e["bottom"] for e in edges if e["orientation"] == "h"]
    if len(raw) < 2:
        return []

    tolerance = _adaptive_tolerance(raw)
    return [_representative(c, spans) for c in _cluster_positions(raw, tolerance)]


def _lattice_tables(page, page_number: int) -> List[Table]:
    """Read cells straight off the page's rules.

    Runs in two passes. The first locates table regions with pdfplumber's own
    line strategy; the second rebuilds each region's grid from that region's
    edges alone, so a dense line-item table and a loose address block on the
    same page each get the snapping they need.
    """
    try:
        found = page.find_tables(table_settings=LATTICE_PROBE)
    except Exception as exc:  # noqa: BLE001
        logger.debug("lattice finder failed on page %s: %s", page_number, exc)
        return []

    tables: List[Table] = []
    for probe in found:
        table = _rebuild_region(page, probe, page_number)
        if table is not None:
            tables.append(table)
    return tables


def _rebuild_region(page, probe, page_number: int) -> Optional[Table]:
    bbox = probe.bbox
    try:
        region = page.crop(bbox, relative=False, strict=False)
        words = region.extract_words(keep_blank_chars=False)
        x_spans = [(float(w["x0"]), float(w["x1"])) for w in words]
        y_spans = [(float(w["top"]), float(w["bottom"])) for w in words]

        xs = _grid_lines(region.edges, "v", x_spans)
        ys = _grid_lines(region.edges, "h", y_spans)
        if len(xs) >= 3 and len(ys) >= 3:
            found = region.find_tables(table_settings={
                "vertical_strategy": "explicit",
                "horizontal_strategy": "explicit",
                "explicit_vertical_lines": xs,
                "explicit_horizontal_lines": ys,
            })
            source = found[0] if found else probe
        else:
            source = probe
        raw = source.extract()
    except Exception as exc:  # noqa: BLE001
        logger.debug("region rebuild failed on page %s: %s", page_number, exc)
        try:
            raw = probe.extract()
        except Exception:  # noqa: BLE001
            return None

    grid = [[_clean_cell(cell) for cell in row] for row in raw]
    grid = [row for row in grid if any(c.strip() for c in row)]
    grid, _ = _drop_empty_columns(grid, None)
    grid = _merge_wrapped_rows(grid)
    if len(grid) < MIN_TABLE_ROWS or max((len(r) for r in grid), default=0) < 2:
        return None

    table = Table(rows=grid, page_numbers=[page_number], source="lines", column_bounds=None)
    setattr(table, "_bbox", bbox)
    return table


def _clean_cell(cell: Optional[str]) -> str:
    """Tidy whitespace but keep the line breaks the cell was laid out with."""
    if not cell:
        return ""
    lines = [" ".join(line.split()) for line in cell.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _outside_bboxes(fragments: Sequence[Fragment], bboxes: Sequence[Tuple]) -> List[Fragment]:
    if not bboxes:
        return list(fragments)
    kept = []
    for frag in fragments:
        inside = any(
            frag.cx >= x0 - 2 and frag.cx <= x1 + 2 and frag.cy >= top - 2 and frag.cy <= bottom + 2
            for x0, top, x1, bottom in bboxes
        )
        if not inside:
            kept.append(frag)
    return kept


# ---------------------------------------------------------------------------
# Scanned pages
# ---------------------------------------------------------------------------


def _render_page(fitz_page, dpi: int = OCR_DPI) -> Tuple[np.ndarray, float]:
    """Rasterise a page. Returns the BGR image and the pixels-per-point zoom."""
    import fitz  # PyMuPDF

    zoom = dpi / 72.0
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:  # PyMuPDF gives RGB; OpenCV and RapidOCR expect BGR
        image = image[:, :, ::-1]
    elif pix.n == 4:
        image = image[:, :, [2, 1, 0]]
    return np.ascontiguousarray(image), zoom


def detect_grid(image: np.ndarray) -> Tuple[List[float], List[float]]:
    """Recover ruled table lines from a scan using morphology.

    Returns the x positions of vertical rules and y positions of horizontal
    rules, in pixels.
    """
    import cv2

    mask = binarize(image)
    h, w = mask.shape[:2]

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 25)))

    horizontal = cv2.dilate(cv2.erode(mask, h_kernel), h_kernel)
    vertical = cv2.dilate(cv2.erode(mask, v_kernel), v_kernel)

    # Erosion has already discarded everything that is not a long run of ink,
    # so the threshold only has to reject residue. It is taken relative to the
    # strongest line rather than to the page: a table's rules are usually far
    # shorter than the page is wide, and an absolute cut would miss them.
    ys = _projection_peaks(horizontal.sum(axis=1) / 255.0, page_extent=w)
    xs = _projection_peaks(vertical.sum(axis=0) / 255.0, page_extent=h)
    return xs, ys


def _projection_peaks(
    profile: np.ndarray, page_extent: float, merge_within: int = 6
) -> List[float]:
    """Collapse runs of strong projection values into single line positions."""
    peak = float(profile.max()) if profile.size else 0.0
    if peak < 0.08 * page_extent:
        return []  # nothing long enough to be a rule
    min_value = max(0.06 * page_extent, 0.35 * peak)

    strong = np.where(profile >= min_value)[0]
    if strong.size == 0:
        return []

    peaks: List[float] = []
    run = [strong[0]]
    for idx in strong[1:]:
        if idx - run[-1] <= merge_within:
            run.append(idx)
        else:
            peaks.append(float(np.mean(run)))
            run = [idx]
    peaks.append(float(np.mean(run)))
    return peaks


def _grid_table_from_ocr(
    fragments: Sequence[Fragment], xs: Sequence[float], ys: Sequence[float], page_number: int
) -> Optional[Table]:
    """Place OCR fragments into a detected line grid."""
    if len(xs) < 3 or len(ys) < 3:
        return None

    xs, ys = sorted(xs), sorted(ys)
    n_cols, n_rows = len(xs) - 1, len(ys) - 1
    if n_cols < 2 or n_rows < MIN_TABLE_ROWS:
        return None

    buckets: Dict[Tuple[int, int], List[Fragment]] = {}
    placed = 0
    for frag in fragments:
        col = _bucket_index(frag.cx, xs)
        row = _bucket_index(frag.cy, ys)
        if col is None or row is None:
            continue
        buckets.setdefault((row, col), []).append(frag)
        placed += 1

    if placed < max(6, 0.35 * len(fragments)):
        return None  # most of the text sits outside the grid; not a real table

    grid = [
        [
            " ".join(f.text for f in sorted(buckets.get((r, c), []), key=lambda f: f.x0)).strip()
            for c in range(n_cols)
        ]
        for r in range(n_rows)
    ]
    grid = [row for row in grid if any(cell.strip() for cell in row)]
    grid, _ = _drop_empty_columns(grid, None)
    if len(grid) < MIN_TABLE_ROWS:
        return None

    return Table(rows=grid, page_numbers=[page_number], source="ocr-grid")


def _bucket_index(value: float, edges: Sequence[float]) -> Optional[int]:
    if value < edges[0] or value > edges[-1]:
        return None
    for i in range(len(edges) - 1):
        if edges[i] <= value <= edges[i + 1]:
            return i
    return None


def _ocr_page_tables(
    fitz_page, page_number: int, ocr: OCREngine
) -> Tuple[List[Table], List[Fragment]]:
    """OCR one page. Returns its tables plus every fragment in PDF-point space."""
    if not ocr.available:
        logger.warning("Page %s looks scanned but no OCR backend is installed", page_number)
        return [], []

    image, zoom = _render_page(fitz_page)
    image = deskew(image)

    fragments_px = ocr.read(image)
    if not fragments_px:
        return [], []
    fragments = [f.scaled(1.0 / zoom) for f in fragments_px]

    xs, ys = detect_grid(image)
    grid_table = _grid_table_from_ocr(fragments_px, xs, ys, page_number)
    if grid_table is not None:
        return [grid_table], fragments

    return _fragments_to_tables(fragments, page_number, source="ocr-text"), fragments


# ---------------------------------------------------------------------------
# Cross-page stitching
# ---------------------------------------------------------------------------


def _normalise(cell: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", cell.lower())


def _row_matches_header(row: Sequence[str], header: Sequence[str]) -> bool:
    """True when a body row is really a repeat of the header."""
    pairs = [
        (_normalise(a), _normalise(b))
        for a, b in zip(row, header)
        if _normalise(a) or _normalise(b)
    ]
    if not pairs:
        return False
    matches = sum(1 for a, b in pairs if a and a == b)
    return matches / len(pairs) >= HEADER_SIMILARITY


def _columns_compatible(a: Table, b: Table) -> bool:
    if a.n_cols != b.n_cols:
        return False
    if not a.column_bounds or not b.column_bounds:
        return True  # line-derived tables: column count is evidence enough
    if len(a.column_bounds) != len(b.column_bounds):
        return False
    return all(abs(p - q) <= 12.0 for p, q in zip(a.column_bounds, b.column_bounds))


def _header_signature(table: Table) -> Optional[Tuple[str, ...]]:
    """A fingerprint of the header row, used to recognise the same table again."""
    if not table.has_header or not table.rows:
        return None
    signature = tuple(_normalise(cell) for cell in table.rows[0])
    named = sum(1 for part in signature if part)
    return signature if named >= max(2, len(signature) // 2) else None


def _can_continue(group: Table, table: Table) -> bool:
    """Is ``table`` a later slice of the same table as ``group``?"""
    if not _columns_compatible(group, table):
        return False
    if max(group.page_numbers) > min(table.page_numbers):
        return False

    group_sig, table_sig = _header_signature(group), _header_signature(table)
    if group_sig and table_sig and group_sig != table_sig:
        return False  # both carry a real header and they disagree: different tables
    return True


def stitch_tables(tables: Sequence[Table]) -> List[Table]:
    """Join tables that continue onto later pages and drop repeated headers.

    Matching looks back over *every* open table, not just the previous one: a
    statement usually sandwiches a page footer or an address block between two
    slices of the same ledger, and those must not break the chain. Otherwise a
    hundred-page statement would land as a hundred sheets.
    """
    for table in tables:
        _mark_header(table)

    groups: List[Table] = []
    for table in tables:
        if not table.rows:
            continue

        target = next((g for g in reversed(groups) if _can_continue(g, table)), None)
        if target is None:
            groups.append(table)
            continue

        body = table.rows
        if target.header and _row_matches_header(body[0], target.header):
            body = body[1:]
        target.rows.extend(body)
        for page in table.page_numbers:
            if page not in target.page_numbers:
                target.page_numbers.append(page)

    # Headers can also reappear mid-table (repeated on every page of a ledger).
    for table in groups:
        if table.header:
            table.rows = [table.rows[0]] + [
                row for row in table.rows[1:] if not _row_matches_header(row, table.header)
            ]

    return _drop_noise_tables([t for t in groups if len(t.rows) >= MIN_TABLE_ROWS])


def _drop_noise_tables(tables: List[Table]) -> List[Table]:
    """Discard stray two-row blocks once the document has a real table.

    Page furniture -- a branch address, an account summary strip -- can look
    tabular in isolation. Next to a 200-row ledger it is clearly not the point
    of the document, and its own sheet just gets in the way.
    """
    if len(tables) < 2:
        return tables
    if not any(len(t.body) >= 5 for t in tables):
        return tables
    return [t for t in tables if len(t.body) >= 2 or t.n_cols >= 4]


def _mark_header(table: Table) -> None:
    """Decide whether the first row is a header or just more data."""
    if not table.rows:
        table.has_header = False
        return
    first = table.rows[0]
    non_empty = [c for c in first if c.strip()]
    if not non_empty:
        table.has_header = False
        return
    numeric_share = sum(1 for c in non_empty if _looks_numeric(c)) / len(non_empty)
    table.has_header = numeric_share < 0.4


# ---------------------------------------------------------------------------
# Document metadata (invoice header fields, statement summary lines, ...)
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r"^\s*([A-Za-z][A-Za-z .,/&'()#-]{2,40}?)\s*[:\-]\s*(.{1,120}?)\s*$")


def extract_key_values(lines: Sequence[str], limit: int = 40) -> List[Tuple[str, str]]:
    """Pull ``Label: value`` pairs out of the prose around the tables."""
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for line in lines:
        for chunk in re.split(r"\s{3,}", line):
            match = _KV_RE.match(chunk)
            if not match:
                continue
            key, value = match.group(1).strip(), match.group(2).strip()
            if not value or len(key) < 3:
                continue
            marker = _normalise(key)
            if marker in seen:
                continue
            seen.add(marker)
            pairs.append((key, value))
            if len(pairs) >= limit:
                return pairs
    return pairs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass
class Document:
    tables: List[Table]
    key_values: List[Tuple[str, str]]
    page_count: int
    scanned_pages: List[int]
    ocr_backend: Optional[str] = None


def extract_document(pdf_bytes: bytes, ocr: Optional[OCREngine] = None) -> Document:
    """Run the full pipeline over a decrypted PDF."""
    import fitz  # PyMuPDF
    import pdfplumber

    ocr = ocr or OCREngine()
    all_tables: List[Table] = []
    prose_lines: List[str] = []
    scanned_pages: List[int] = []
    used_ocr = False

    fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for index, page in enumerate(pdf.pages):
                page_number = index + 1
                fragments = _page_words(page)

                if len(fragments) < MIN_WORDS_DIGITAL:
                    scanned_pages.append(page_number)
                    tables, ocr_fragments = _ocr_page_tables(fitz_doc[index], page_number, ocr)
                    used_ocr = used_ocr or bool(ocr_fragments)
                    all_tables.extend(tables)
                    prose_lines.extend(_page_prose(ocr_fragments, tables))
                    continue

                page_tables: List[Table] = []
                bboxes: List[Tuple] = []
                if _has_ruling_lines(page):
                    page_tables = _lattice_tables(page, page_number)
                    bboxes = [getattr(t, "_bbox") for t in page_tables if hasattr(t, "_bbox")]

                remaining = _outside_bboxes(fragments, bboxes)
                page_tables.extend(_fragments_to_tables(remaining, page_number, "text"))

                if not page_tables:
                    # A text layer can exist yet hold no tables (e.g. a scanned
                    # page with a thin OCR stamp) -- retry through OCR.
                    fallback, ocr_fragments = _ocr_page_tables(fitz_doc[index], page_number, ocr)
                    if fallback:
                        scanned_pages.append(page_number)
                        used_ocr = True
                        page_tables = fallback
                        fragments = ocr_fragments

                all_tables.extend(page_tables)
                prose_lines.extend(_page_prose(fragments, page_tables))

            page_count = len(pdf.pages)
    finally:
        fitz_doc.close()

    all_tables.sort(key=lambda t: (min(t.page_numbers), 0))
    stitched = stitch_tables(all_tables)

    return Document(
        tables=stitched,
        key_values=extract_key_values(prose_lines),
        page_count=page_count,
        scanned_pages=scanned_pages,
        ocr_backend=ocr.name if used_ocr else None,
    )


def _line_text(row: Sequence[Fragment], wide_gap: float) -> str:
    """Join a line, widening the separator where the layout leaves a real gap.

    The run of spaces is what later lets ``extract_key_values`` tell
    ``Invoice No: 871`` and ``GSTIN: 27AAB...`` apart when they sit side by side.
    """
    ordered = sorted(row, key=lambda f: f.x0)
    parts: List[str] = []
    for i, frag in enumerate(ordered):
        if i:
            parts.append("   " if frag.x0 - ordered[i - 1].x1 > wide_gap else " ")
        parts.append(frag.text)
    return "".join(parts).strip()


def _page_prose(fragments: Sequence[Fragment], tables: Sequence[Table]) -> List[str]:
    """Lines on the page that are not part of any table -- invoice headers etc."""
    table_text = {
        _normalise(cell) for table in tables for row in table.rows for cell in row if cell.strip()
    }
    rows = cluster_rows(fragments)
    wide_gap = max(6.0, 1.2 * _median_height(rows))

    lines = []
    for row in rows:
        text = _line_text(row, wide_gap)
        if text and _normalise(text) not in table_text:
            lines.append(text)
    return lines
