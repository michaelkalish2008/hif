"""SessionEngine: load once, profile many.

The execution core shared by `hif profile` (one prompt), `hif batch`
(a workload file), `hif study` (sessions of workloads), and — later — a
local daemon (`hif serve`). Model, embedder, teacher-forcing surrogate,
and analyzer weights are loaded exactly once per engine instance; each
`profile_one` call then pays inference cost only.

The engine never writes anything implicitly: persisting an artifact is the
caller's explicit act via `write_trace`. It follows the pipeline's compute-
and-discard default for the raw variant and branch traces, which is now an
artifact-size decision rather than a data-handling one — see
`hif/metrics/field.py` for what compute-and-discard still buys and what it no
longer has to claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hif.config import ModelConfig, RunConfig
from hif.profile.record import profile_hash, signals_record

# Default teacher-forcing surrogate: ungated mirror of Llama 3.2 1B (same
# weights as meta-llama/Llama-3.2-1B), so runs aren't blocked on HF gated-repo
# access. Mirrors the study harness and the CLI --surrogate default.
DEFAULT_SURROGATE_MODEL_ID = "unsloth/Llama-3.2-1B"


class SessionEngine:
    """Holds loaded models and profiles prompts against them.

    Construct via `SessionEngine.create(config, ...)` for the standard
    loading path, or pass pre-loaded components (tests, harnesses that manage
    their own models).
    """

    def __init__(self, config: RunConfig, model, embedder,
                 surrogate_model=None):
        self.config = config
        self.model = model
        self.embedder = embedder
        self.surrogate_model = surrogate_model

    # -- construction --------------------------------------------------------

    @classmethod
    def create(
        cls,
        config: RunConfig,
        *,
        surrogate: bool = False,
        surrogate_model_id: str = DEFAULT_SURROGATE_MODEL_ID,
    ) -> "SessionEngine":
        """Load model + embedder (+ surrogate when requested AND needed).

        A surrogate is only loaded when the target backend cannot teacher-
        force — on backends that can, input-side signals come from the target
        itself and `surrogate` is ignored (same rule as the CLI flag).
        """
        from hif.clustering.embed import EmbeddingModel
        from hif.models.factory import load_model

        model = load_model(config.model)
        embedder = EmbeddingModel(config.embedding)

        surrogate_model = None
        if surrogate and not model.supports_teacher_forcing:
            surrogate_model = load_model(ModelConfig(
                name=surrogate_model_id, backend="hf",
                device="auto", dtype="bfloat16",
            ))
        return cls(config, model, embedder, surrogate_model)

    # -- profiling -----------------------------------------------------------

    def profile_one(
        self,
        prompt,
        *,
        regime: str = "ordinary_conversation",
        seed: int = 42,
        authored_variants: Optional[list] = None,
        variant_output_sink: Optional[dict] = None,
    ):
        """Run the full pipeline on one prompt.

        Returns the in-memory BehavioralRangeProfile. Nothing is written to
        disk — see `write_trace` for the explicit persistence opt-in.

        `authored_variants`: researcher-authored perturbation texts for THIS
        prompt; when given they replace the configured generator pipeline.
        Resolved by the caller (a workload row's `variants`, or the CLI
        reading [perturbation] variants_file) — the builder never does file
        I/O, so what it measured is exactly what it was handed.

        `variant_output_sink`: a caller-owned dict that receives
        {variant_text: variant_continuation} for every variant continuation
        the run elicits. Out-of-band on purpose: the record's `variant_io`
        block needs the texts on request (--variant-io), but persisting them
        on the profile would grow the artifact's content surface for every
        run, opted in or not. A sink the caller passes is content the caller
        asked to hold.
        """
        from hif.profile.builder import build_profile

        return build_profile(
            self.model, prompt, regime, self.config, self.embedder, seed,
            surrogate_model=self.surrogate_model,
            authored_variants=authored_variants,
            variant_output_sink=variant_output_sink,
        )

    def record_for(
        self,
        profile,
        *,
        prompt: str,
        regime: str,
        seed: int,
        latency: Optional[dict] = None,
        trace_path: Optional[str] = None,
        extras: Optional[dict] = None,
        include_units: bool = False,
    ) -> dict:
        """The canonical derived-signals record for a profile from this engine."""
        return signals_record(
            profile,
            model_name=self.config.model.name,
            backend=self.config.model.backend,
            regime=regime,
            seed=seed,
            prompt=prompt,
            latency=latency,
            trace_path=trace_path,
            extras=extras,
            include_units=include_units,
        )

    # -- traceability (explicit opt-in) --------------------------------------

    def write_trace(self, profile, *, prompt: str, seed: int,
                    trace_dir: Path) -> Path:
        """Persist the full profile artifact for traceability/accountability.

        Called when the run opted in (RunConfig.traceability.enabled /
        --trace), which is what puts the raw variant and branch traces in the
        artifact. The baseline per-step top-K is in the JSON either way — it
        lives on `profile.output_side.steps`, so the flag was never the line
        between "distributions on disk" and not, whatever the old wording
        implied. Returns the artifact path (hash-addressed, content-stable).
        """
        from hif.profile.render_json import render_json

        trace_dir = Path(trace_dir)
        h = profile_hash(self.config.model.name, prompt, seed)
        path = trace_dir / f"profile_{h}.json"
        render_json(profile, path)
        return path
