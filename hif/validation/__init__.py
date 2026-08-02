"""hif.validation — known-answer validation assets.

The product carries its own evidence: the synthetic known-answer corpus and
the region-sensitivity harness here back CI integration tests, per-model
support gating for VLM adapters, compression validation reuse, and the
future `hif validate-model` CLI command. The multimodal_v1 study
(scripts/study/run_multimodal_v1.py) is the first execution of this
capability. (The originating validation-loop spec is not public; the
behaviour is defined by this package and its tests.)
"""

from hif.validation.corpus import (  # noqa: F401
    GRID_ROWS,
    GRID_COLS,
    CELL,
    IMG_SIZE,
    PILOT_IMAGE_IDS,
    assert_bbox_in_cell,
    cell_rect,
    generate_corpus,
    load_corpus,
)
from hif.validation.harness import (  # noqa: F401
    ValidationResult,
    validate_region_sensitivity,
)
