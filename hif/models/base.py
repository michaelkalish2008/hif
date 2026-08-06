"""Abstract base class defining the model interface all backends must satisfy."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pydantic import BaseModel


class Logits(BaseModel):
    """Full logit tensor at every position in the input sequence.

    Shape: (seq_len, vocab_size). Stored as a Python list of lists
    so it serializes cleanly; callers convert to numpy as needed.
    """

    values: list[list[float]]  # shape (seq_len, vocab_size)
    seq_len: int
    vocab_size: int

    def to_numpy(self) -> np.ndarray:
        return np.array(self.values, dtype=np.float32)


class TopKEntry(BaseModel):
    """One candidate token at a single generation step."""

    token_id: int
    token_str: str
    logit: float
    logprob: float
    prob: float


class StepRecord(BaseModel):
    """Everything recorded at one generation step."""

    step: int
    selected_token_id: int
    selected_token_str: str
    topk: list[TopKEntry]  # length == K


class GenerationResult(BaseModel):
    """Output of model.generate()."""

    input_ids: list[int]
    generated_ids: list[int]  # new tokens only
    steps: list[StepRecord]  # one per generated token
    model_name: str
    top_k: int
    seed: int


class Model(ABC):
    """Abstract interface for a model under analysis.

    Exposes only the external behavior of the model: tokenization,
    full-vocabulary logits (teacher-forcing), and top-K generation.
    Never exposes activations, attention, or any internal state.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def context_length(self) -> int: ...

    @property
    @abstractmethod
    def max_top_k(self) -> Optional[int]:
        """None means unlimited (HF/TLens); 20 for Ollama."""

    @property
    @abstractmethod
    def supports_teacher_forcing(self) -> bool:
        """True for HF and TLens; False for Ollama."""

    @abstractmethod
    def tokenize(self, text: str) -> list[int]: ...

    @abstractmethod
    def detokenize(self, ids: list[int]) -> str: ...

    @abstractmethod
    def forward(self, input_ids: list[int]) -> Logits:
        """Run teacher-forced forward pass. Returns logits at every position.

        Raises NotImplementedError on backends that do not support full
        teacher forcing (e.g., OllamaModel).
        """

    @abstractmethod
    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        top_k: int,
        seed: int,
    ) -> GenerationResult:
        """Generate text while collecting top-K logits at each step."""
