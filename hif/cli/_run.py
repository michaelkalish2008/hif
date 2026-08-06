"""The shared pipeline call.

`profile` and `batch` both run the same pipeline; this is the one place that
call is written, so the two commands cannot drift into two behaviours.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from hif.cli._app import (
    console,
)
from hif.cli._config import (
    _make_run_config,
)
from hif.cli._load import (
    _load_surrogate,
)
# Called through their modules, not bound by name at import: a test that
# patches `hif.cli._load._load_model` must affect every caller, and
# `from ... import _load_model` would bind a copy this module keeps using.
from hif.cli import _load

# Canonical measurement extraction lives in hif/profile/measure.py so the CLI
# tables and the SessionEngine record path report identical numbers; the
# registry rows they are keyed on live in hif/profile/registry.py and the wire
# record in hif/profile/record.py.
from hif.profile.measure import (
    measurements as _measurements,
    prompt_measurements as _prompt_measurements,
)
from hif.profile.record import (
    RECORD_SCHEMA_VERSION as SIGNAL_RECORD_VERSION,
    profile_hash as _profile_hash,
    signals_record as _signals_record,
)
from hif.profile.registry import (
    run_subjects as _run_subjects,
)



# ---------------------------------------------------------------------------
# The shared pipeline call
# ---------------------------------------------------------------------------


def _resolve_run_config(
    model_name: str,
    backend: str,
    max_new_tokens: int,
    top_k: int,
    seed: int,
    output_dir: Optional[Path],
    *,
    diagnostics: bool = False,
    base_config=None,
    explicit: frozenset = frozenset(),
    n_perturbation_variants: int = 2,
    trace: bool = False,
    lite: bool = False,
    acquisition: str = "elicited-output",
):
    """Every source of configuration, resolved in one place.

    This is the SINGLE resolution path: `hif profile` runs whatever this
    returns, and `hif config show` prints whatever this returns, so the two
    can never drift. Precedence, later beats earlier:

      schema defaults → --config-file → --mode/--diagnostics →
      explicit CLI flags → --acquisition → --lite

    --acquisition and --lite apply last because they are ceilings: a run
    asking for less must never silently do more, regardless of what a config
    file switched on.
    """
    config = _make_run_config(model_name, backend, max_new_tokens, top_k, seed, output_dir,
                               diagnostics=diagnostics, base=base_config,
                               explicit=explicit)
    # Apply perturbation variant count from --mode — unless a --config-file
    # set its own perturbation budget and the user didn't pass --mode.
    if base_config is None or "mode" in explicit:
        config.perturbation.n_variants = n_perturbation_variants
    config.traceability.enabled = trace

    # --acquisition is a CEILING on what the run may bring into existence, not
    # a speed knob. It is applied before --lite because the two answer different
    # questions and compose: this one says what the run is permitted to produce,
    # --lite says how much work to do within that permission.
    if acquisition == "observational":
        # Nothing beyond the caller's own call. No authored prompts, no
        # elicited continuations, no branches. variants_file too: the
        # researcher wrote those strings, but sending them is still more than
        # the one call the tier permits.
        config.perturbation.generators = []
        config.perturbation.n_variants = 0
        config.perturbation.variants_file = None
        config.trajectory.n_branches = 0
    elif acquisition == "synthesized-input":
        # Paraphrases are authored and teacher-forced; the model never
        # generates from them, and branches are elicitation by definition.
        config.perturbation.elicit_variant_outputs = False
        config.trajectory.n_branches = 0

    # --lite drops every stage that costs an extra generation pass or an
    # embedding sweep: the paraphrase variants, the trajectory branches, and
    # the per-step candidate geometry (which exposure reads). What survives is
    # the single baseline pass plus input-side teacher forcing, so the
    # entropy-side measurements come back unchanged and the rest come back
    # ABSENT — omitted from the record rather than reported as zero. It applies
    # last so it beats a --config-file that switched these stages on; a run
    # asking for less should never silently do more.
    if lite:
        config.perturbation.generators = []
        config.perturbation.n_variants = 0
        config.perturbation.variants_file = None
        config.trajectory.n_branches = 0
        config.semantic.enabled = False
        config.exposure.enabled = False
        config.semantic_field.enabled = False
        config.attention.enabled = False

    return config


def _run_single_profile(
    model_name: str,
    prompt: str,
    regime: str,
    backend: str,
    seed: int,
    output_dir: Optional[Path],
    max_new_tokens: int,
    top_k: int,
    charts: bool = False,
    model=None,
    embedder=None,
    n_perturbation_variants: int = 2,
    diagnostics: bool = False,
    metric: Optional[str] = None,
    surrogate: bool = False,
    surrogate_model_id: str = "unsloth/Llama-3.2-1B",
    trace: bool = False,
    trace_dir: Optional[Path] = None,
    base_config=None,
    explicit: frozenset = frozenset(),
    lite: bool = False,
    acquisition: str = "elicited-output",
    variant_output_sink: Optional[dict] = None,
) -> "tuple[BehavioralRangeProfile, Optional[Path]]":
    """Core pipeline: build the profile in memory, return (profile, trace_path).

    Privacy-first default: NOTHING is written to disk. `output_dir` opts into
    the derived reports (markdown, charts); `trace` opts into persisting the
    full profile artifact (raw per-step top-K distributions) for
    traceability/reconstruction — trace_path is None unless it did.
    """
    from hif.engine import SessionEngine
    from hif.profile.render_markdown import render_public, render_technical

    config = _resolve_run_config(
        model_name, backend, max_new_tokens, top_k, seed, output_dir,
        diagnostics=diagnostics, base_config=base_config, explicit=explicit,
        n_perturbation_variants=n_perturbation_variants, trace=trace,
        lite=lite, acquisition=acquisition,
    )

    if model is None:
        model = _load._load_model(model_name, backend)
    if embedder is None:
        embedder = _load._load_embedder()

    surrogate_model = None
    if surrogate:
        if model.supports_teacher_forcing:
            console.print(
                "  [dim]--surrogate ignored: this backend already teacher-forces "
                "(input-side signals come from the target itself).[/dim]"
            )
        else:
            surrogate_model = _load_surrogate(surrogate_model_id)

    engine = SessionEngine(config, model, embedder, surrogate_model)

    # Researcher-authored variants: resolved HERE, not in the builder, from
    # the workload-format JSONL the config points at. The builder is handed
    # texts, so what it measured is exactly what the caller resolved.
    authored_variants: Optional[list] = None
    if config.perturbation.variants_file is not None:
        from hif.perturbation.authored import load_authored_variants

        authored_variants = load_authored_variants(
            config.perturbation.variants_file, prompt
        )

    profile = engine.profile_one(prompt, regime=regime, seed=seed,
                                 authored_variants=authored_variants,
                                 variant_output_sink=variant_output_sink)

    h = _profile_hash(model_name, prompt, seed)

    trace_path: Optional[Path] = None
    if trace:
        resolved_trace_dir = trace_dir or (
            (output_dir / "traces") if output_dir else Path("traces")
        )
        trace_path = engine.write_trace(profile, prompt=prompt, seed=seed,
                                        trace_dir=resolved_trace_dir)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        render_technical(profile, output_dir / f"profile_{h}_technical.md")
        render_public(profile, output_dir / f"profile_{h}_public.md")

    if charts and output_dir is not None:
        from hif.viz import generate_signal_plots
        plots_dir = output_dir / "plots" / h
        if metric is not None:
            # Single metric → a single chart that gives the printed number meaning.
            res = generate_signal_plots(profile, plots_dir, only_signal=metric)
            chart = res.get(metric, {}).get("html")
            if chart is not None:
                console.print(f"  [dim]Chart: {chart}[/dim]")
        else:
            res = generate_signal_plots(profile, plots_dir)
            index = res.get("index", {}).get("html")
            if index is not None:
                console.print(f"  [dim]Dashboard: {index}[/dim]")

    return profile, trace_path


# ---------------------------------------------------------------------------
# Commands — Local plane
# ---------------------------------------------------------------------------


