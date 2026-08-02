"""Abstract bases for perturbation: text generators and media families.

Text generators (`PerturbationGenerator`, str -> str) are the original
namespace, resolved via `get_generator()`. Media-side perturbation
(Design §6, docs/ARCHITECTURE.md § Multimodal notes) uses the separate
`PerturbationFamily` protocol,
resolved via `get_family()` — the two namespaces never mix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hif.models.mm import MultimodalInput


class PerturbationResult(BaseModel):
    original: str
    variants: list[str]
    generator: str  # name of the generator that produced these


class PerturbationGenerator(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        """Generate n_variants perturbed versions of prompt."""


# ---------------------------------------------------------------------------
# Media-side perturbation (Design §6, docs/ARCHITECTURE.md § Multimodal notes)
# ---------------------------------------------------------------------------


class PerturbationTrace(BaseModel):
    """What a media family did to which part — the only persisted record.

    Never carries pixels: regions/params are geometry and knobs only.
    """

    family: str                       # e.g. "image_grid_mask"
    part_index: int                   # which MultimodalInput part was perturbed
    regions: list[dict] = Field(default_factory=list)  # M1: [{"row": r, "col": c}]
    params: dict[str, Any] = Field(default_factory=dict)


class MultimodalVariant(BaseModel):
    """One perturbed input. `input` is a NEW object; the original is never mutated."""

    model_config = {"arbitrary_types_allowed": True}

    input: Any                        # runtime type: MultimodalInput (lazy to avoid import cycle)
    trace: PerturbationTrace


class PerturbationFamily(ABC):
    """Media-side perturbation protocol (parallel to PerturbationGenerator).

    supported_kinds is the set of InputPart kinds this family can perturb
    ({"image"} in M1). perturb() must return variants whose inputs are new
    MultimodalInput objects; perturbed media live as in-memory image_bytes
    parts and are never written to disk or serialized.
    """

    name: str
    supported_kinds: set[str]

    @abstractmethod
    def perturb(
        self, mm_input: "MultimodalInput", n_variants: int, seed: int
    ) -> list[MultimodalVariant]:
        """Produce up to n_variants perturbed copies of mm_input.

        Convention: n_variants <= 0 means exhaustive sweep (audit mode) for
        families with a finite region set; fast mode passes a positive count.
        """
