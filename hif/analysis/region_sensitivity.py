"""Region-sensitivity artifact: per-grid-cell perturbation JSD.

Assembled from (PerturbationTrace.regions, SensitivityMetrics) pairs produced
by the image_grid_mask family (MULTIMODAL.md § Design §3/§6). This is
perturbation-JSD only — it never touches generation-model attention (Risk
rule 7).

Copy rule (Risk rules 7-8): human-facing strings describe cells whose masking
"materially affected the model's response behavior" — no causal, correctness,
or attention language.

Runtime-typed on the profile (BehavioralRangeProfile.region_sensitivity is
Optional[Any], lazy like `attention`).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from hif.metrics.sensitivity import SensitivityMetrics
from hif.perturbation.base import PerturbationTrace


class RegionCell(BaseModel):
    row: int
    col: int
    jsd: float  # mean JS divergence of variants masking this cell


class RegionSensitivityResult(BaseModel):
    """Per-cell mean perturbation JSD for one image part's mask grid."""

    part_index: int
    grid_rows: int
    grid_cols: int
    cells: list[RegionCell]
    max_cell: RegionCell | None = None  # cell with the highest mean JSD
    mean_jsd: float = 0.0               # mean over measured cells

    def to_text_grid(self) -> str:
        """ASCII heatmap for CLI output.

        Values are per-cell mean JS divergence under grid masking — higher
        values mark regions whose masking materially affected the model's
        response behavior. Unmeasured cells (fast mode) render as "  .  ".
        """
        lookup = {(c.row, c.col): c.jsd for c in self.cells}
        lines = [
            f"Region sensitivity (part {self.part_index}, "
            f"{self.grid_rows}x{self.grid_cols} grid, mean JSD per masked cell):"
        ]
        for r in range(self.grid_rows):
            row_vals = []
            for c in range(self.grid_cols):
                v = lookup.get((r, c))
                row_vals.append("  .  " if v is None else f"{v:.3f}")
            lines.append("  " + "  ".join(row_vals))
        if self.max_cell is not None:
            lines.append(
                f"  Masking cell (row {self.max_cell.row}, col "
                f"{self.max_cell.col}) most materially affected the model's "
                f"response behavior (JSD {self.max_cell.jsd:.3f})."
            )
        return "\n".join(lines)


def assemble_region_sensitivity(
    pairs: list[tuple[PerturbationTrace, SensitivityMetrics]],
) -> RegionSensitivityResult | None:
    """Build the artifact from grid-mask (trace, sensitivity) pairs.

    Only single-region grid-mask traces contribute (M1: one cell per
    variant). Multiple variants hitting the same cell are averaged. Returns
    None when no grid-mask pairs exist. M1 assumes a single image part; the
    first grid-mask part_index seen wins.
    """
    grid_pairs = [
        (t, s)
        for t, s in pairs
        if t.family == "image_grid_mask" and t.regions
    ]
    if not grid_pairs:
        return None

    part_index = grid_pairs[0][0].part_index
    grid_rows = int(grid_pairs[0][0].params.get("grid_rows", 0))
    grid_cols = int(grid_pairs[0][0].params.get("grid_cols", 0))

    per_cell: dict[tuple[int, int], list[float]] = {}
    for trace, sens in grid_pairs:
        if trace.part_index != part_index:
            continue
        for region in trace.regions:
            key = (int(region["row"]), int(region["col"]))
            per_cell.setdefault(key, []).append(sens.mean_js_divergence)

    cells = [
        RegionCell(row=r, col=c, jsd=float(np.mean(vals)))
        for (r, c), vals in sorted(per_cell.items())
    ]
    max_cell = max(cells, key=lambda c: c.jsd) if cells else None
    mean_jsd = float(np.mean([c.jsd for c in cells])) if cells else 0.0

    return RegionSensitivityResult(
        part_index=part_index,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        cells=cells,
        max_cell=max_cell,
        mean_jsd=mean_jsd,
    )
