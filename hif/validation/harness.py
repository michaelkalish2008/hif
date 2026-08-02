"""Region-sensitivity validation harness (product asset).

Runs the known-answer corpus against a multimodal model: for each image ×
question variant, sweeps a grid mask over the image, measures JSD between the
baseline output distribution and each masked variant's (the Sensitivity
family), and ranks the KNOWN answer cell against all other cells.

Acceptance criterion: the answer cell ranks in the top 2 of grid cells for at
least `threshold` (default 70%) of image × variant combinations.

This harness backs:
  (a) a CI integration test,
  (b) per-model support gating — a VLM adapter is not "supported" until it
      passes this suite,
  (c) compression-validation reuse (same corpus + JSD machinery),
  (d) the future `hif validate-model` CLI command.

The multimodal_v1 study is the first execution of this capability.

Inference path (M1 session 2, live): each image × question is profiled via
`build_profile` with the image_grid_mask family in audit mode (exhaustive
cell sweep) at the harness grid size; per-cell JSD comes from the
RegionSensitivityResult artifact on the profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hif.validation.corpus import load_corpus

DEFAULT_THRESHOLD = 0.70
DEFAULT_GRID = (4, 4)
DEFAULT_MAX_NEW_TOKENS = 24

_EMBEDDER = None  # module-level cache; drivers/tests may inject directly


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from hif.clustering.embed import EmbeddingModel
        from hif.config import EmbeddingConfig

        _EMBEDDER = EmbeddingModel(EmbeddingConfig())
    return _EMBEDDER


@dataclass
class ImageValidationRecord:
    """Per image × variant outcome of the region-sensitivity sweep."""
    image_id: str
    variant: int
    question: str
    answer_cell: dict          # {"row": r, "col": c}
    cell_jsd: dict             # {"r,c": mean JSD vs baseline} — one entry per grid cell
    answer_cell_rank: int      # 1 = answer cell had the highest JSD of all cells
    in_top2: bool


@dataclass
class ValidationResult:
    model_id: str
    grid: tuple[int, int]
    threshold: float
    top2_rate: float
    passed: bool
    per_image: list[ImageValidationRecord] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"region-sensitivity validation [{status}] model={self.model_id} "
                f"grid={self.grid[0]}x{self.grid[1]} top2_rate={self.top2_rate:.2f} "
                f"(threshold {self.threshold:.2f}, n={len(self.per_image)})")


def validate_region_sensitivity(
    model,
    corpus_dir: Path | str | None = None,
    grid: tuple[int, int] = DEFAULT_GRID,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    pilot: bool = False,
    seed: int = 0,
) -> ValidationResult:
    """Run the known-answer region-sensitivity suite against `model`.

    Returns a ValidationResult with pass/fail, the top-2 rate, and per-image
    records. Passing this suite is the acceptance gate for a VLM adapter.
    """
    records = load_corpus(corpus_dir, pilot=pilot)
    per_image: list[ImageValidationRecord] = []

    for rec in records:
        answer_cell = scale_cell(rec["answer_cell"], rec.get("grid"), grid)
        for variant, question in enumerate(rec["question_variants"]):
            cell_jsd = _sweep_image(model, rec, question, grid, seed)
            rank = compute_answer_cell_rank(cell_jsd, answer_cell)
            per_image.append(ImageValidationRecord(
                image_id=rec["image_id"], variant=variant, question=question,
                answer_cell=answer_cell, cell_jsd=cell_jsd,
                answer_cell_rank=rank, in_top2=rank <= 2,
            ))

    top2_rate = (sum(1 for r in per_image if r.in_top2) / len(per_image)) if per_image else 0.0
    return ValidationResult(
        model_id=getattr(model, "model_id", str(model)),
        grid=grid,
        threshold=threshold,
        top2_rate=top2_rate,
        passed=top2_rate >= threshold,
        per_image=per_image,
    )


def scale_cell(cell: dict, corpus_grid: dict | None,
               grid: tuple[int, int]) -> dict:
    """Map a corpus answer cell (declared on the corpus's own grid, e.g. 4x4)
    onto the harness grid (e.g. 2x2 in the pilot). Identity when grids match.

    The corpus guarantees the answer bbox lies fully inside its declared
    cell, so any coarser grid that nests the corpus grid preserves the
    fully-inside invariant after scaling.
    """
    rows, cols = grid
    src_rows = int((corpus_grid or {}).get("rows", rows))
    src_cols = int((corpus_grid or {}).get("cols", cols))
    return {
        "row": int(cell["row"]) * rows // src_rows,
        "col": int(cell["col"]) * cols // src_cols,
    }


def compute_answer_cell_rank(cell_jsd: dict[str, float], answer_cell: dict) -> int:
    """Rank of the answer cell by descending JSD (rank 1 = highest JSD).

    `cell_jsd` maps "r,c" -> mean JSD; `answer_cell` is {"row": r, "col": c}
    on the SAME grid as cell_jsd. Raises KeyError if the answer cell was not
    measured (audit mode always measures every cell).
    """
    key = f"{answer_cell['row']},{answer_cell['col']}"
    if key not in cell_jsd:
        raise KeyError(f"answer cell {key} not present in cell_jsd {sorted(cell_jsd)}")
    ranked = sorted(cell_jsd.items(), key=lambda kv: kv[1], reverse=True)
    return next(i for i, (k, _) in enumerate(ranked, start=1) if k == key)


def _build_mm_input(record: dict, question: str):
    """Image part + question text part for one corpus record."""
    from hif.models.mm import InputPart, MultimodalInput

    return MultimodalInput(parts=[
        InputPart.from_image_path(record["path"]),
        InputPart.from_text(f"Question: {question} Answer:"),
    ])


def _profile_for(model, record: dict, question: str, grid: tuple[int, int],
                 seed: int, embedder=None,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
    """Run build_profile on one image × question with the image_grid_mask
    family in audit mode (exhaustive cell sweep) at the harness grid size."""
    from hif.config import RunConfig
    from hif.profile.builder import build_profile

    rows, cols = grid
    config = RunConfig()
    config.perturbation.media_families = ["image_grid_mask"]
    config.perturbation.image_grid_rows = rows
    config.perturbation.image_grid_cols = cols
    config.perturbation.n_variants = 0  # audit mode: exhaustive sweep
    config.generation.max_new_tokens = max_new_tokens

    return build_profile(
        model,
        _build_mm_input(record, question),
        regime="validation",
        config=config,
        embedder=embedder if embedder is not None else _get_embedder(),
        seed=seed,
    )


def _sweep_image(model, record: dict, question: str, grid: tuple[int, int],
                 seed: int) -> dict[str, float]:
    """Baseline + full grid-mask sweep for one image × question.

    Returns {"r,c": mean JSD vs baseline}, extracted from the profile's
    RegionSensitivityResult artifact.
    """
    profile = _profile_for(model, record, question, grid, seed)
    rs = profile.region_sensitivity
    if rs is None or not rs.cells:
        raise RuntimeError(
            f"No region_sensitivity artifact produced for "
            f"{record['image_id']!r} (question={question!r}) — check that the "
            "image_grid_mask family ran."
        )
    return {f"{c.row},{c.col}": float(c.jsd) for c in rs.cells}
