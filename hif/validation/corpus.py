"""Synthetic known-answer corpus generator (product asset).

Generates 900×900px synthetic form and chart images aligned to a 4×4 grid
(225px cells), where the answer-bearing element of each image sits fully
inside a KNOWN grid cell. Deterministic from a seed. Used by:

  - the multimodal_v1 validation study (scripts/study/*)
  - the region-sensitivity validation harness (hif/validation/harness.py)
  - future CI integration tests and `hif validate-model`

Every generated image is verified in code: the answer text bounding box is
checked to lie fully within the declared answer cell (raises CellBoundsError
otherwise).

PIL-only — no model inference happens here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

IMG_SIZE = 900
GRID_ROWS = 4
GRID_COLS = 4
CELL = IMG_SIZE // GRID_ROWS  # 225

DEFAULT_SEED = 20260703

# Pilot subset: 2 forms + 2 charts
PILOT_IMAGE_IDS = ["form_01", "form_02", "chart_01", "chart_02"]


class CellBoundsError(ValueError):
    """An answer-bearing element was not fully inside its declared grid cell."""


def _font(size: int) -> ImageFont.ImageFont:
    """Deterministic cross-platform font: Pillow's bundled default at a size."""
    return ImageFont.load_default(size=size)


def cell_rect(row: int, col: int, cell_px: int = CELL) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) pixel rect of a grid cell."""
    return (col * cell_px, row * cell_px, (col + 1) * cell_px, (row + 1) * cell_px)


def assert_bbox_in_cell(
    bbox: tuple[float, float, float, float],
    row: int,
    col: int,
    label: str,
    cell_px: int = CELL,
) -> None:
    """Raise CellBoundsError unless bbox lies fully inside the (row, col) cell."""
    x0, y0, x1, y1 = cell_rect(row, col, cell_px)
    bx0, by0, bx1, by1 = bbox
    if not (bx0 >= x0 and by0 >= y0 and bx1 <= x1 and by1 <= y1):
        raise CellBoundsError(
            f"{label}: answer bbox {bbox} not fully inside declared cell "
            f"(row={row}, col={col}) rect {cell_rect(row, col, cell_px)}"
        )


def _draw_text_in_cell(
    draw: ImageDraw.ImageDraw,
    text: str,
    row: int,
    col: int,
    font: ImageFont.ImageFont,
    fill: str = "black",
    y_frac: float = 0.5,
) -> tuple[float, float, float, float]:
    """Draw text centered horizontally inside a cell at vertical fraction y_frac,
    shrinking the font (with an 8px margin) until it fits the cell width.
    Returns the drawn text bbox (checked by callers to be inside the cell)."""
    x0, y0, x1, y1 = cell_rect(row, col)
    size = getattr(font, "size", 24)
    while size > 10:
        tb = draw.textbbox((0, 0), text, font=font)
        if (tb[2] - tb[0]) <= (x1 - x0) - 16:
            break
        size -= 2
        font = _font(size)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = x0 + ((x1 - x0) - tw) / 2 - tb[0]
    ty = y0 + ((y1 - y0) - th) * y_frac - tb[1]
    draw.text((tx, ty), text, font=font, fill=fill)
    return draw.textbbox((tx, ty), text, font=font)


# ---------------------------------------------------------------------------
# Content pools (deterministic; indexed, not sampled from live state)
# ---------------------------------------------------------------------------

FORM_SPECS = [
    {
        "image_id": "form_01",
        "title": "INVOICE #4417",
        "critical": ("Total:", "$4,832"),
        "question": "What is the total amount on the invoice?",
        "question_variants": [
            "What is the total amount on the invoice?",
            "how much is the invoice total?",
            "Please state the total amount shown on this invoice document.",
        ],
        "expected_answer": "$4,832",
        "distractors": [("Invoice date:", "2026-03-14"), ("Customer ID:", "CX-2291"),
                        ("Subtotal:", "$4,410"), ("Tax:", "$422"), ("Terms:", "Net 30")],
    },
    {
        "image_id": "form_02",
        "title": "MEDICATION ORDER",
        "critical": ("Dosage:", "10 mg/kg"),
        "question": "What is the prescribed dosage?",
        "question_variants": [
            "What is the prescribed dosage?",
            "what dosage does the order say?",
            "Please report the dosage specified on this medication order form.",
        ],
        "expected_answer": "10 mg/kg",
        "distractors": [("Patient ID:", "P-88410"), ("Drug:", "Cefalexin"),
                        ("Route:", "Oral"), ("Frequency:", "BID"), ("Prescriber:", "Dr. Osei")],
    },
    {
        "image_id": "form_03",
        "title": "SHIPPING MANIFEST",
        "critical": ("Weight:", "412 kg"),
        "question": "What is the shipment weight?",
        "question_variants": [
            "What is the shipment weight?",
            "how heavy is the shipment?",
            "Please state the total weight recorded on this shipping manifest.",
        ],
        "expected_answer": "412 kg",
        "distractors": [("Manifest #:", "SM-70233"), ("Origin:", "Rotterdam"),
                        ("Destination:", "Oslo"), ("Pieces:", "18"), ("Carrier:", "NordFreight")],
    },
    {
        "image_id": "form_04",
        "title": "LAB REQUISITION",
        "critical": ("Glucose:", "142 mg/dL"),
        "question": "What is the reported glucose value?",
        "question_variants": [
            "What is the reported glucose value?",
            "what's the glucose reading on the form?",
            "Please report the glucose value documented on this laboratory requisition.",
        ],
        "expected_answer": "142 mg/dL",
        "distractors": [("Accession #:", "L-55102"), ("Collected:", "07:40"),
                        ("Sodium:", "139 mmol/L"), ("Potassium:", "4.1 mmol/L"), ("Fasting:", "Yes")],
    },
    {
        "image_id": "form_05",
        "title": "EXPENSE CLAIM",
        "critical": ("Reimbursement:", "$1,275"),
        "question": "What is the reimbursement amount?",
        "question_variants": [
            "What is the reimbursement amount?",
            "how much is being reimbursed?",
            "Please state the reimbursement amount entered on this expense claim.",
        ],
        "expected_answer": "$1,275",
        "distractors": [("Claim #:", "EC-3308"), ("Employee:", "R. Tanaka"),
                        ("Period:", "May 2026"), ("Mileage:", "180 mi"), ("Approved by:", "K. Ellis")],
    },
]

CHART_SPECS = [
    {
        "image_id": "chart_01",
        "title": "Units Sold by Product",
        "labels": ["Alpha", "Bravo", "Cedar", "Delta"],
        "winner": "Cedar",
        "question": "Which product sold the most units?",
        "question_variants": [
            "Which product sold the most units?",
            "which product was the top seller?",
            "Please identify the product with the highest unit sales in this chart.",
        ],
    },
    {
        "image_id": "chart_02",
        "title": "Support Tickets by Region",
        "labels": ["North", "South", "East", "West"],
        "winner": "East",
        "question": "Which region has the most support tickets?",
        "question_variants": [
            "Which region has the most support tickets?",
            "which region filed the most tickets?",
            "Please identify the region with the greatest number of support tickets shown.",
        ],
    },
    {
        "image_id": "chart_03",
        "title": "Downloads by Platform",
        "labels": ["iOS", "Android", "Web", "Desktop"],
        "winner": "Android",
        "question": "Which platform has the most downloads?",
        "question_variants": [
            "Which platform has the most downloads?",
            "what platform got downloaded the most?",
            "Please identify the platform with the highest download count in this chart.",
        ],
    },
    {
        "image_id": "chart_04",
        "title": "Energy Use by Building",
        "labels": ["Annex", "Tower", "Lab", "Depot"],
        "winner": "Tower",
        "question": "Which building uses the most energy?",
        "question_variants": [
            "Which building uses the most energy?",
            "which building is the biggest energy user?",
            "Please identify the building with the highest energy use shown in this chart.",
        ],
    },
    {
        "image_id": "chart_05",
        "title": "Applications by Department",
        "labels": ["Sales", "Legal", "R&D", "Ops"],
        "winner": "R&D",
        "question": "Which department received the most applications?",
        "question_variants": [
            "Which department received the most applications?",
            "which department got the most applications?",
            "Please identify the department with the highest application count in this chart.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_form(spec: dict, rng: random.Random) -> tuple[Image.Image, dict]:
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    body_font = _font(24)
    title_font = _font(30)

    # Title band along the very top (outside any content claim)
    draw.text((30, 12), spec["title"], font=title_font, fill="black")
    draw.line((30, 56, IMG_SIZE - 30, 56), fill="black", width=2)

    # Choose the answer cell (avoid row 0 so the title band never collides)
    answer_row = rng.randint(1, GRID_ROWS - 1)
    answer_col = rng.randint(0, GRID_COLS - 1)

    # Distractor cells: distinct cells != answer cell, avoiding the row-0 title band
    candidates = [(r, c) for r in range(1, GRID_ROWS) for c in range(GRID_COLS)
                  if (r, c) != (answer_row, answer_col)]
    rng.shuffle(candidates)
    distractor_cells = candidates[: len(spec["distractors"])]

    # Critical field, fully inside its cell
    label, value = spec["critical"]
    bbox = _draw_text_in_cell(draw, f"{label} {value}", answer_row, answer_col, body_font)
    assert_bbox_in_cell(bbox, answer_row, answer_col, spec["image_id"])

    # Distractor fields
    for (r, c), (dl, dv) in zip(distractor_cells, spec["distractors"]):
        db = _draw_text_in_cell(draw, f"{dl} {dv}", r, c, body_font)
        assert_bbox_in_cell(db, r, c, f"{spec['image_id']} distractor {dl}")

    meta = {
        "answer_cell": {"row": answer_row, "col": answer_col},
        "distractor_cells": [{"row": r, "col": c} for r, c in distractor_cells],
        "text_twin": _form_text_twin(spec),
    }
    return img, meta


def _form_text_twin(spec: dict) -> str:
    lines = [spec["title"]]
    fields = list(spec["distractors"])
    label, value = spec["critical"]
    fields.insert(len(fields) // 2, (label, value))  # bury the critical field mid-list
    return "\n".join(lines + [f"{fl} {fv}" for fl, fv in fields])


def _render_chart(spec: dict, rng: random.Random) -> tuple[Image.Image, dict]:
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    label_font = _font(24)
    title_font = _font(30)

    draw.text((30, 12), spec["title"], font=title_font, fill="black")
    baseline_y = 860
    draw.line((30, baseline_y, IMG_SIZE - 30, baseline_y), fill="black", width=2)

    labels = spec["labels"]
    winner_idx = labels.index(spec["winner"])

    # Answer cell: the tallest bar's label sits fully inside this cell.
    # Rows 0–2 are reachable by a bar top (row 3 holds the axis); the column is
    # the bar's column (4 bars ↔ 4 grid columns).
    answer_row = rng.randint(0, 2)
    answer_col = winner_idx

    # Bar geometry: one bar per grid column, 130px wide, centered in its column.
    bar_w = 130
    heights: list[int] = []
    # Non-winner bar tops land strictly below the answer cell.
    min_loser_top = (answer_row + 1) * CELL + 30
    for i in range(len(labels)):
        if i == winner_idx:
            continue
        heights.append(rng.randint(60, max(70, baseline_y - min_loser_top - 40)))
    rng.shuffle(heights)

    tops: dict[int, int] = {}
    h_iter = iter(heights)
    for i in range(len(labels)):
        if i == winner_idx:
            continue
        tops[i] = baseline_y - next(h_iter)

    # Winner: label centered inside the answer cell, bar top just below the label.
    label_text = spec["winner"]
    lbx0, lby0, lbx1, lby1 = _draw_text_in_cell(
        draw, label_text, answer_row, answer_col, label_font, y_frac=0.35
    )
    assert_bbox_in_cell((lbx0, lby0, lbx1, lby1), answer_row, answer_col, spec["image_id"])
    tops[winner_idx] = int(lby1) + 8

    for i, name in enumerate(labels):
        x_center = i * CELL + CELL // 2
        top = tops[i]
        draw.rectangle((x_center - bar_w // 2, top, x_center + bar_w // 2, baseline_y),
                       outline="black", width=2,
                       fill=(200, 210, 235) if i == winner_idx else (225, 225, 225))
        if i != winner_idx:
            # Loser labels above their bars (never inside the answer cell by construction)
            tb = draw.textbbox((0, 0), name, font=label_font)
            tw = tb[2] - tb[0]
            draw.text((x_center - tw / 2, top - 34), name, font=label_font, fill="black")

    # Sanity: winner bar is strictly the tallest
    winner_h = baseline_y - tops[winner_idx]
    if not all(baseline_y - t < winner_h for i, t in tops.items() if i != winner_idx):
        raise CellBoundsError(f"{spec['image_id']}: winner bar is not strictly tallest")

    distractor_cells = []
    for i in range(len(labels)):
        if i == winner_idx:
            continue
        r = min(GRID_ROWS - 1, tops[i] // CELL)
        distractor_cells.append({"row": int(r), "col": i})

    twin_lines = [spec["title"]] + [
        f"{name}: {baseline_y - tops[i]} units" for i, name in enumerate(labels)
    ]
    meta = {
        "answer_cell": {"row": answer_row, "col": answer_col},
        "distractor_cells": distractor_cells,
        "text_twin": "\n".join(twin_lines),
    }
    return img, meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_corpus(seed: int = DEFAULT_SEED, out_dir: Path | str | None = None,
                    pilot: bool = False) -> list[dict]:
    """Generate the known-answer corpus and write images + corpus.jsonl.

    Deterministic from `seed`; per-image RNG streams mean the --pilot subset
    renders images identical to the full run. Returns the corpus records.
    """
    if out_dir is None:
        out_dir = default_corpus_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    specs = [("form", s) for s in FORM_SPECS] + [("chart", s) for s in CHART_SPECS]
    for kind, spec in specs:
        if pilot and spec["image_id"] not in PILOT_IMAGE_IDS:
            continue
        img_rng = random.Random(f"{seed}:{spec['image_id']}")
        if kind == "form":
            img, meta = _render_form(spec, img_rng)
            expected = spec["expected_answer"]
        else:
            img, meta = _render_chart(spec, img_rng)
            expected = spec["winner"]

        path = out_dir / f"{spec['image_id']}.png"
        img.save(path)

        records.append({
            "image_id": spec["image_id"],
            "path": str(path),
            "kind": kind,
            "question": spec["question"],
            "question_variants": spec["question_variants"],
            "expected_answer": expected,
            "answer_cell": meta["answer_cell"],
            "distractor_cells": meta["distractor_cells"],
            "text_twin": meta["text_twin"],
            "grid": {"rows": GRID_ROWS, "cols": GRID_COLS, "cell_px": CELL},
            "pilot": spec["image_id"] in PILOT_IMAGE_IDS,
            "seed": seed,
        })

    with open(out_dir / "corpus.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return records


def default_corpus_dir() -> Path:
    """Repo-relative default: data/studies/multimodal_v1/corpus/."""
    return (Path(__file__).parent.parent.parent
            / "data" / "studies" / "multimodal_v1" / "corpus")


def load_corpus(corpus_dir: Path | str | None = None, pilot: bool = False) -> list[dict]:
    """Load corpus.jsonl records; optionally only the pilot subset."""
    corpus_dir = Path(corpus_dir) if corpus_dir else default_corpus_dir()
    jsonl = corpus_dir / "corpus.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"Corpus not found at {jsonl}.")
    records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    if pilot:
        records = [r for r in records if r.get("pilot")]
    return records
