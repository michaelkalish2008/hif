"""Abstract bases for perturbation.

`PerturbationGenerator` (str -> str) is resolved via `get_generator()`.
There was a second, media-side namespace (`PerturbationFamily`, resolved by
`get_family()`); it went with the image path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class PerturbationResult(BaseModel):
    original: str
    variants: list[str]
    generator: str  # name of the generator that produced these


class PerturbationGenerator(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        """Generate n_variants perturbed versions of prompt."""
