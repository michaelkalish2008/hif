"""HI command-line interface."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Canonical measurement extraction lives in hif/profile/signals.py so the CLI
# tables and the SessionEngine record path report identical numbers.
from hif.profile.signals import (
    MEASUREMENT_KEYS,
    MEASUREMENT_REGISTRY,
    RECORD_SCHEMA_VERSION as SIGNAL_RECORD_VERSION,
    MEASUREMENT_UNITS,
    RESOLUTIONS,
    SIGNAL_SET_VERSION,
    SUBJECT_LEGEND,
    SUBJECT_MIXED,
    SUBJECT_PROMPT_ONLY,
    measurements as _measurements,
    profile_hash as _profile_hash,
    prompt_measurements as _prompt_measurements,
    run_subjects as _run_subjects,
    signals_record as _signals_record,
)

app = typer.Typer(
    name="hif",
    help="Horizonal Interpretability — using the horizon of the possibility space to "
    "describe model behaviour. Every measurement is reported in its natural unit "
    "(bits, cosine distance, Pearson r, a fraction of steps); nothing is normalised, "
    "inverted, or thresholded. Run `hif schema` for the full measurement set.",
)
# stdout is reserved for data. Every human-facing line — progress, warnings,
# tables, errors — goes to stderr so `hif <cmd> ... | jq .` always parses.
console = Console(stderr=True)
err_console = console


def _emit_json_line(record: dict) -> None:
    """Write one JSONL record to stdout and flush.

    stdout carries JSON and nothing else. Every data-producing command uses
    this (or a single json.dumps for the one-document commands), so
    `hif <cmd> ... 2>/dev/null | jq .` always parses.
    """
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Sub-apps for command groups
# ---------------------------------------------------------------------------

@app.callback()
def _main() -> None:
    """hif — Horizonal Interpretability CLI."""
    from hif.utils.logging import configure_logging

    # Default: results only. Commands that accept --verbose re-call
    # configure_logging(verbose=True) to restore full internal chatter.
    configure_logging(verbose=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_config_file(path: Path) -> "RunConfig":
    """Parse a TOML --config-file into a RunConfig (pydantic-validated).

    Table names mirror RunConfig fields ([generation], [perturbation],
    [attention], [semantic_field], [trajectory], ...). Exit 3 on parse or
    validation errors — a half-applied config silently changing what the
    numbers mean is worse than no run at all.
    """
    import tomllib
    from hif.config import RunConfig

    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError:
        err_console.print(f"[red]--config-file not found: {path}[/red]")
        raise typer.Exit(3)
    except tomllib.TOMLDecodeError as exc:
        err_console.print(f"[red]Could not parse --config-file {path}: {exc}[/red]")
        raise typer.Exit(3)
    # RunConfig tolerates unknown fields (forward compatibility for embedded
    # profile JSON), so a typo'd table ([perturbaton]) would be silently
    # dropped here — reject unknown top-level keys explicitly instead.
    unknown = sorted(set(data) - set(RunConfig.model_fields))
    if unknown:
        err_console.print(
            f"[red]Unknown key(s) in --config-file {path}: "
            f"{', '.join(unknown)}. "
            f"Valid tables: {', '.join(sorted(RunConfig.model_fields))}.[/red]"
        )
        raise typer.Exit(3)
    try:
        return RunConfig(**data)
    except Exception as exc:
        err_console.print(f"[red]Invalid --config-file {path}: {exc}[/red]")
        raise typer.Exit(3)


def _make_run_config(
    model_name: str,
    backend: str,
    max_new_tokens: int,
    top_k: int,
    seed: int,
    output_dir: Optional[Path],
    diagnostics: bool = False,
    base: "Optional[RunConfig]" = None,
    explicit: frozenset = frozenset(),
) -> "RunConfig":
    """Assemble the RunConfig for a profile run.

    `base` is a TOML-loaded RunConfig (--config-file); when present it wins
    for everything EXCEPT the model identity (always from the CLI args) and
    any generation knob the user passed explicitly on the command line
    (`explicit` holds those parameter names, from typer's parameter sources).
    --diagnostics only ever turns analyzers ON — it never disables one a
    config file enabled.

    Temperature precedence: the sampling adapters consume
    ModelConfig.temperature (not GenerationConfig.temperature), so a
    [generation] temperature set in the TOML is mirrored onto
    cfg.model.temperature here. An explicit [model] temperature in the TOML
    wins over the mirror; when neither was set, model.temperature stays None
    (each backend's own default — 0 for OpenAI, unchanged sampling for HF).
    GenerationConfig.temperature defaults to 1.0, so the mirror fires only
    when the TOML actually set it (model_fields_set), never off the default —
    mirroring the 1.0 default would silently change API-backend behavior.
    """
    from hif.config import (
        AttentionConfig,
        GenerationConfig,
        ModelConfig,
        OutputConfig,
        RunConfig,
        SemanticFieldConfig,
    )

    if base is not None:
        cfg = base.model_copy(deep=True)
        cfg.model = ModelConfig(name=model_name, backend=backend)
        # Model identity (name/backend) always comes from the CLI args (see
        # docstring), but a [model] base_url/api_key/dtype in the TOML — the
        # only way to point an "openai"-backend arm at an OpenAI-compatible
        # endpoint (Mistral, DeepSeek, Grok, local/vLLM) — has to survive the
        # ModelConfig replacement above or the request silently goes to the
        # real OpenAI API instead, asking it for a model name it's never
        # heard of (404 "model does not exist").
        if "base_url" in base.model.model_fields_set:
            cfg.model.base_url = base.model.base_url
        if "api_key" in base.model.model_fields_set:
            cfg.model.api_key = base.model.api_key
        if "dtype" in base.model.model_fields_set:
            cfg.model.dtype = base.model.dtype
        if "revision" in base.model.model_fields_set:
            cfg.model.revision = base.model.revision
        # Temperature plumbing (see docstring): [model] temperature wins;
        # otherwise mirror an explicitly-set [generation] temperature onto
        # the model config the sampling adapters actually read.
        if "temperature" in base.model.model_fields_set:
            cfg.model.temperature = base.model.temperature
        elif "temperature" in base.generation.model_fields_set:
            cfg.model.temperature = cfg.generation.temperature
        if "max_new_tokens" in explicit:
            cfg.generation.max_new_tokens = max_new_tokens
        if "top_k" in explicit:
            cfg.generation.top_k = top_k
        if "seed" in explicit:
            cfg.generation.seed = seed
        if output_dir is not None:
            cfg.output.output_dir = output_dir
        if diagnostics:
            cfg.attention.enabled = True
            cfg.semantic_field.enabled = True
        return cfg

    return RunConfig(
        model=ModelConfig(name=model_name, backend=backend),
        generation=GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            seed=seed,
        ),
        # output_dir=None means "write nothing" (privacy-first default); the
        # OutputConfig still needs a placeholder path — nothing consults it
        # unless the CLI explicitly writes reports/charts under --output-dir.
        output=OutputConfig(output_dir=output_dir or Path("outputs")),
        # Spread/Horizon (Instrument readings) come from an independent
        # DistilBERT text analyzer — backend-agnostic, so it's worth the
        # extra load only when --diagnostics will actually show the readings.
        attention=AttentionConfig(enabled=diagnostics),
        # Veer (semantic field) re-embeds each step's candidate cloud —
        # enabled under --diagnostics alongside the other instrument readings.
        semantic_field=SemanticFieldConfig(enabled=diagnostics),
    )


def _load_dotenv() -> None:
    """Load .env from the repo root (walks up from cwd). Silent if python-dotenv absent."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)  # searches cwd upward; don't clobber shell env vars
    except ImportError:
        pass


def _load_model(model_name: str, backend: str):
    _load_dotenv()
    from hif.config import ModelConfig
    # Ollama-style names ("gemma3:4b-it-qat") contain a colon, which is invalid
    # in HuggingFace repo ids — auto-route to the ollama backend rather than
    # failing with an obscure repo-id validation error.
    if backend == "hf" and ":" in model_name:
        err_console.print(
            f"[yellow]{model_name!r} looks like an Ollama model tag — using "
            "--backend ollama. Pass --backend explicitly to override.[/yellow]"
        )
        backend = "ollama"
    from hif.models.factory import load_model
    try:
        return load_model(ModelConfig(name=model_name, backend=backend))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def _load_embedder():
    from hif.clustering.embed import EmbeddingModel
    from hif.config import EmbeddingConfig
    return EmbeddingModel(EmbeddingConfig())


# Absent-signal rendering.
#
# StabilityMetrics components are Optional: None means the signal is ABSENT —
# not measurable for this run (e.g. partial-access API backends have no
# input-side series), which is deliberately distinct from a pinned/degenerate
# value (the multimodal_v1 H1 defect, since fixed by the absent-not-pinned
# rule in hif/metrics/stability.py). The CLI renders absent signals as
# "n/a", never as numbers; non-None values print normally on text and
# multimodal runs alike.
# Absent means "this run produced no evidence for that quantity" — a backend
# that cannot teacher-force, an analysis stage that did not run. It is
# deliberately distinct from a measured value and is never rendered as a
# number.
ABSENT_TEXT = "absent (not measurable on this backend/run)"


def _profile_modality(p) -> str:
    return getattr(p.prompt, "modality", "text") or "text"


def _build_multimodal_input(image_paths: list[Path], prompt: str):
    """Validate image files and assemble a MultimodalInput (images first, then
    text — matching the multimodal_v1 study construction). Exit 3 on any
    unreadable/non-image file."""
    from hif.models.mm import InputPart, MultimodalInput

    parts = []
    for path in image_paths:
        if not path.exists():
            err_console.print(f"[red]--input file not found: {path}[/red]")
            raise typer.Exit(3)
        try:
            from PIL import Image

            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            err_console.print(
                f"[red]--input {path} is not a readable image (PNG/JPEG): {exc}[/red]"
            )
            raise typer.Exit(3)
        parts.append(InputPart.from_image_path(str(path)))
    parts.append(InputPart.from_text(prompt))
    return MultimodalInput(parts=parts)


def _signal_set_family(version: str) -> str:
    """Major family of a signal-set version: "hif-v1.1" -> "hif-v1".

    Versions within one family are additive supersets: comparison proceeds
    over the intersection of measurements present in both artifacts, naming
    each exclusion. Different families are a true mismatch.
    """
    m = re.match(r"^(.*-v\d+)", version or "")
    return m.group(1) if m else (version or "")


def _artifact_signal_set_version(data: dict) -> str:
    """Signal-set version recorded on a profile/baseline/prior JSON dict.

    Priors and baselines record `protocol_version`; hosted profiles record
    `signal_set_version`. Artifacts predating both read as "hif-v1"."""
    return data.get("signal_set_version") or data.get("protocol_version") or "hif-v1"


def _signal_set_mismatch_exit(baseline_version: str, candidate_version: str) -> None:
    """Different major signal-set families: hard error, exit 2 (mirrors the
    platform 409). Same-family minor differences never reach here — they
    compare over the intersection instead."""
    err_console.print(
        f"[red]These artifacts were scored under different signal sets "
        f'("{baseline_version}" vs "{candidate_version}"). Re-profile them '
        "under the same HIF Signal Set version to compare.[/red]"
    )
    raise typer.Exit(2)


def _modality_mismatch_exit(baseline_modality: str, candidate_modality: str) -> None:
    """Cross-modality comparison is a different experimental condition, not a
    difference in the model — hard error, exit 2."""
    err_console.print(
        f"[red]A {baseline_modality} profile is a different experimental "
        f"condition than a {candidate_modality} profile. Re-profile both "
        "under the same modality to compare.[/red]"
    )
    raise typer.Exit(2)


def _load_surrogate(model_id: str):
    """Load a small open-weight model to teacher-force the prompt+output when the
    target backend can't (hosted APIs, Ollama).

    Recovers the input-side signals (Stability, Surprise, I/O Correlation, Wager)
    the target cannot expose — the same teacher-forcing "proxy" the study harness
    uses. Defaults to Llama 3.2 1B (ungated mirror)."""
    from hif.config import ModelConfig
    from hif.models.hf import HFModel

    console.print(f"  [dim]Loading teacher-forcing surrogate: {model_id}…[/dim]")
    return HFModel(ModelConfig(
        name=model_id, backend="hf", device="auto", dtype="bfloat16",
    ))


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
    mm_input=None,
    diagnostics: bool = False,
    metric: Optional[str] = None,
    surrogate: bool = False,
    surrogate_model_id: str = "unsloth/Llama-3.2-1B",
    trace: bool = False,
    trace_dir: Optional[Path] = None,
    base_config=None,
    explicit: frozenset = frozenset(),
) -> "tuple[BehavioralRangeProfile, Optional[Path]]":
    """Core pipeline: build the profile in memory, return (profile, trace_path).

    Privacy-first default: NOTHING is written to disk. `output_dir` opts into
    the derived reports (markdown, charts); `trace` opts into persisting the
    full profile artifact (raw per-step top-K distributions) for
    traceability/reconstruction — trace_path is None unless it did.

    When mm_input is provided (a MultimodalInput with media parts), it is
    passed to build_profile in place of the text prompt; `prompt` still names
    the text portion for hashing/labels.
    """
    from hif.engine import SessionEngine
    from hif.profile.render_markdown import render_public, render_technical

    config = _make_run_config(model_name, backend, max_new_tokens, top_k, seed, output_dir,
                               diagnostics=diagnostics, base=base_config,
                               explicit=explicit)
    # Apply perturbation variant count from --mode — unless a --config-file
    # set its own perturbation budget and the user didn't pass --mode.
    if base_config is None or "mode" in explicit:
        config.perturbation.n_variants = n_perturbation_variants
    config.traceability.enabled = trace

    if model is None:
        model = _load_model(model_name, backend)
    if embedder is None:
        embedder = _load_embedder()

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
    profile = engine.profile_one(mm_input if mm_input is not None else prompt,
                                 regime=regime, seed=seed)

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


@app.command()
def profile(
    ctx: typer.Context,
    model_name: str = typer.Argument(..., help="Model name (e.g. gpt2)"),
    prompt: str = typer.Argument(..., help="Prompt text"),
    regime: str = typer.Option("ordinary_conversation", help="Prompt regime"),
    backend: str = typer.Option(
        "hf",
        help="Model backend: hf | tlens | ollama | openai | anthropic | gemini | "
        "hf-vlm | openai-vlm. For image inputs (--input) use an explicit VLM "
        "backend: hf-vlm (local AutoModelForImageTextToText checkpoints, e.g. "
        "SmolVLM/Gemma 3 multimodal) or openai-vlm (hosted vision API, e.g. gpt-4o).",
    ),
    input_files: list[Path] = typer.Option(
        [],
        "--input",
        help="Image file (PNG/JPEG) to include as model input; repeatable. "
        "Images are presented before the prompt text. Requires --backend "
        "hf-vlm or openai-vlm.",
    ),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Write derived reports (technical + public markdown, --charts "
        "plots) here. Default: nothing is written to disk — results print to "
        "the terminal only (privacy-first compute-and-discard).",
    ),
    max_new_tokens: int = typer.Option(64, help="Maximum new tokens to generate"),
    top_k: int = typer.Option(50, help="Top-K candidates per step"),
    config_file: Optional[Path] = typer.Option(
        None,
        help="TOML run config (tables mirror RunConfig: [generation], "
        "[perturbation], [trajectory], [attention], [semantic_field], ...). "
        "CLI flags you pass explicitly override the file.",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="Opt-in traceability: persist the full profile artifact (raw "
        "per-step top-K distributions — reconstructable content) so signals "
        "can be recomputed or audited later without re-running the model. "
        "Default off: compute-and-discard.",
    ),
    trace_dir: Optional[Path] = typer.Option(
        None,
        "--trace-dir",
        help="Where --trace artifacts are written (default: <output-dir>/traces, "
        "or ./traces when no --output-dir). Passing this implies --trace.",
    ),
    charts: bool = typer.Option(
        False,
        "--charts",
        help="Generate plots + the combined dashboard locally (off by default).",
    ),
    diagnostics: bool = typer.Option(
        False,
        "--diagnostics",
        help="Also run the two optional analysis stages — attention capture "
        "(the two attention-row entropy measurements) and the semantic field "
        "(centroid veer). Both cost extra compute, so they are off by default; "
        "the measurements they produce are simply absent from the record when "
        "they have not run.",
    ),
    application: Optional[str] = typer.Option(
        None,
        help="Application archetype (support-chatbot, rag-qa, coding-assistant, summarization, extraction, classification, agent-tool-use, multimodal-qa, document-understanding). "
        "Selects the perturbation family and the default analysis window.",
    ),
    mode: str = typer.Option(
        "fast",
        help="fast: fewer perturbation variants, smaller default analysis window. "
        "audit: full perturbation set, governance-ready report. "
        "Input is always passed in full regardless of mode.",
    ),
    analysis_window: Optional[str] = typer.Option(
        None,
        help="Maximum output tokens to analyze (does not truncate inference). "
        "Integer or 'adaptive' (default: adaptive = analyze all output).",
    ),
    metric: Optional[str] = typer.Option(
        None,
        help="Print ONE measurement, in its natural unit, and exit. "
        "Run `hif schema` for the full list with unit definitions.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show model input/output text, perturbation variants, full numeric stats, "
        "effective-config notes, and full internal logging (pipeline + HTTP chatter)",
    ),
    output_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON profile"),
    units: bool = typer.Option(
        False, "--units",
        help="Include a per-measurement units block in each record. Constant per "
             "signal_set_version and identical on every record, so off by default; "
             "`hif schema` prints the same information without running a model.",
    ),
    truncate: Optional[int] = typer.Option(
        None,
        "--truncate",
        help="Truncate input to N tokens before analysis. Results reflect truncated context only.",
    ),
    surrogate: bool = typer.Option(
        False,
        "--surrogate",
        help="Recover the input-side signals (stability, surprise, io_correlation, "
        "wager) on backends that cannot teacher-force (ollama, openai, gemini, "
        "anthropic) by teacher-forcing a small local proxy model over the "
        "prompt+output — the same technique the study harness uses. Ignored when the "
        "target backend already teacher-forces (hf/tlens/hf-vlm). Implied by "
        "--surrogate-model, so passing that alone is enough.",
    ),
    surrogate_model: Optional[str] = typer.Option(
        None,
        "--surrogate-model",
        help="Open-weight HF model id used for --surrogate (default: Llama 3.2 1B, "
        "ungated mirror). Passing this flag implies --surrogate — you don't need both.",
    ),
) -> None:
    """Run the full HI pipeline on a single (model, prompt) pair."""
    # --surrogate-model implies --surrogate: passing a model id but forgetting
    # the boolean flag used to silently do nothing (input-side signals still
    # zeroed with a warning).
    if surrogate_model is not None:
        surrogate = True
    effective_surrogate_model = surrogate_model or "unsloth/Llama-3.2-1B"

    if verbose:
        from hif.utils.logging import configure_logging

        configure_logging(verbose=True)

    base_config = _load_config_file(config_file) if config_file is not None else None
    # Which options the user passed explicitly (vs. their defaults) — these
    # override --config-file values in _make_run_config.
    # Compare by enum NAME, not identity/equality: typer >=0.26 returns its
    # own ParameterSource enum class rather than click's, so a cross-class
    # `!=` against click.core.ParameterSource.DEFAULT is always True.
    explicit = frozenset(
        name for name in ("max_new_tokens", "top_k", "seed", "mode")
        if (src := ctx.get_parameter_source(name)) is not None
        and getattr(src, "name", None) != "DEFAULT"
    )

    # A --config-file [generation] seed wins over the CLI *default* (an
    # explicit --seed still beats the file) — the seed used for the run,
    # hashing, and labels must match config.generation.seed.
    if base_config is not None and "seed" not in explicit:
        seed = base_config.generation.seed

    # --trace-dir implies --trace (passing a destination but forgetting the
    # boolean should not silently discard the artifacts you asked for).
    if trace_dir is not None:
        trace = True

    if charts and output_dir is None:
        err_console.print(
            "[red]--charts writes plot files — pass --output-dir to say "
            "where.[/red]"
        )
        raise typer.Exit(3)

    if mode not in ("fast", "audit"):
        err_console.print(f"[red]--mode must be 'fast' or 'audit', got {mode!r}[/red]")
        raise typer.Exit(3)

    if metric is not None and metric not in MEASUREMENT_KEYS:
        err_console.print(
            f"[red]--metric must be one of: {', '.join(MEASUREMENT_KEYS)} — "
            f"got {metric!r}[/red]"
        )
        raise typer.Exit(3)

    # Capability guard: fail fast when the requested metric can't be produced by
    # the chosen backend, BEFORE loading the model or running the pipeline.
    # (This is what would have caught `--metric stability --backend ollama`.)
    if metric is not None:
        # Mirror the colon auto-route so the guard reflects the real backend.
        _effective_backend = "ollama" if (backend == "hf" and ":" in model_name) else backend
        from hif.models.capabilities import metric_support
        # The attention rows are gated on the analysis STAGE, not the backend
        # (nothing reads the target's attention). --diagnostics turns it on;
        # so can a --config file, which is why both are consulted here.
        _attention_on = bool(diagnostics) or bool(
            base_config is not None and base_config.attention.enabled
        )
        # A teacher-forcing surrogate changes what a restricted backend can
        # produce (input-side rows, and the candidate-cloud rows it rebuilds
        # from the target's actual continuation), so the guard is asked the
        # question the run will actually face. Which surrogate recovery applies
        # to which measurement is capabilities.py's to know, not the CLI's —
        # this used to be an ad-hoc exemption here for the input-side set only,
        # which silently refused `--metric output_entropy_bits --surrogate` on
        # a selected-only backend that produces it perfectly well.
        _reason = metric_support(
            metric, _effective_backend,
            attention_enabled=_attention_on, surrogate=bool(surrogate),
        )
        if _reason is not None:
            err_console.print(
                f"[red]Cannot compute --metric {metric} on --backend "
                f"{_effective_backend}.[/red]\n[yellow]{_reason}[/yellow]\n"
                "[dim]Run `hif models` to see which measurements each backend "
                "supports.[/dim]"
            )
            raise typer.Exit(3)

    # Multimodal input: validate image files and assemble the input up front,
    # before any model work (exit 3 on bad files).
    mm_input = None
    run_modality = "text"
    if input_files:
        if backend not in ("hf-vlm", "openai-vlm"):
            err_console.print(
                "[red]--input requires a multimodal backend: use --backend "
                "hf-vlm (local VLM) or --backend openai-vlm (hosted vision API).[/red]"
            )
            raise typer.Exit(3)
        mm_input = _build_multimodal_input(list(input_files), prompt)
        run_modality = mm_input.modality

    # mode affects perturbation count only; input is always full unless --truncate is set
    if mm_input is not None:
        # Multimodal: audit = exhaustive grid-mask sweep (n_variants<=0 means
        # every cell, 16 on the default 4x4 grid); fast = default variant budget.
        n_variants = 0 if mode == "audit" else 2
    else:
        n_variants = 2 if mode == "fast" else 5

    # Input truncation — explicit user choice only, never a silent default
    input_truncated = False
    if truncate is not None:
        if mm_input is not None:
            err_console.print(
                "[red]--truncate is not supported with --input (image runs "
                "always send the full input — media inputs are the "
                "experimental condition).[/red]"
            )
            raise typer.Exit(3)
        if truncate <= 0:
            err_console.print("[red]--truncate must be a positive integer[/red]")
            raise typer.Exit(3)
        # Approximate token truncation by whitespace-split words (no tokenizer dep here)
        words = prompt.split()
        if len(words) > truncate:
            prompt = " ".join(words[:truncate])
            input_truncated = True
            err_console.print(
                f"[yellow]Warning: Input truncated to {truncate} tokens — "
                "results reflect truncated context only.[/yellow]"
            )

    # Validate --application against the archetype registry and apply defaults
    archetype_def = None
    if application:
        from hif.archetypes import UnknownArchetypeError, load_archetype

        try:
            archetype_def = load_archetype(application)
        except UnknownArchetypeError as exc:
            err_console.print(
                f"[red]Unknown archetype {application!r}. "
                f"Valid: {', '.join(exc.valid_ids)}[/red]"
            )
            raise typer.Exit(3)
        # Apply the archetype's default analysis window when not set by the user
        if analysis_window is None:
            analysis_window = str(archetype_def.default_analysis_window)

    # Resolve analysis_window
    analysis_window_val: Optional[int] = None
    if analysis_window and analysis_window != "adaptive":
        try:
            analysis_window_val = int(analysis_window)
        except ValueError:
            err_console.print(f"[red]--analysis-window must be an integer or 'adaptive', got {analysis_window!r}[/red]")
            raise typer.Exit(3)

    if not output_json:
        console.print(f"[bold]HI Profile[/bold]")
        if application:
            console.print(f"  Application: {application}")
        console.print(f"  Model:   {model_name} ({backend})")
        console.print(f"  Prompt:  {prompt[:80]!r}")
        if input_files:
            for f in input_files:
                console.print(f"  Input:   {f}")
            console.print(f"  Modality: {run_modality}")
        # In single-profile mode the regime is a category label only — it tags
        # the run (run_id) but does not change any metric computation
        # and nothing is compared against it. Say so, so the number isn't
        # mistaken for a regime-relative score.
        console.print(f"  Regime:  {regime} [dim](label only — not a comparison)[/dim]")
        console.print(f"  Seed:    {seed}")
        console.print(f"  Mode:    {mode}")
        if analysis_window:
            console.print(f"  Analysis window: {analysis_window}")
        console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        timings: dict[str, float] = {}
        t_total = time.perf_counter()
        try:
            t0 = time.perf_counter()
            model = _load_model(model_name, backend)
            timings["model_load"] = time.perf_counter() - t0
            progress.update(task, description="Loading embedder...")
            t0 = time.perf_counter()
            embedder = _load_embedder()
            timings["embedder_load"] = time.perf_counter() - t0
            progress.update(task, description="Running pipeline...")
            t0 = time.perf_counter()
            p, trace_path = _run_single_profile(
                model_name=model_name,
                prompt=prompt,
                regime=regime,
                backend=backend,
                seed=seed,
                output_dir=output_dir,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                charts=charts,
                model=model,
                embedder=embedder,
                n_perturbation_variants=n_variants,
                mm_input=mm_input,
                diagnostics=diagnostics,
                metric=metric,
                surrogate=surrogate,
                surrogate_model_id=effective_surrogate_model,
                trace=trace,
                trace_dir=trace_dir,
                base_config=base_config,
                explicit=explicit,
            )
            timings["pipeline"] = time.perf_counter() - t0
            progress.update(task, description="Done.")
        except Exception as exc:
            err_console.print(f"[red]Pipeline error: {exc}[/red]")
            raise typer.Exit(1)
        timings["total"] = time.perf_counter() - t_total

    h = _profile_hash(model_name, prompt, seed)

    # --metric: print one measurement, in its natural unit, and exit.
    if metric is not None:
        from hif.profile.signals import _prompt_reference_model

        subject = _run_subjects(p).get(metric)
        prompt_only = subject == SUBJECT_PROMPT_ONLY
        vals = _prompt_measurements(p) if prompt_only else _measurements(p)
        value = vals.get(metric)
        if value is None:
            err_console.print(
                f"[red]--metric {metric}: {ABSENT_TEXT}.[/red]\n"
                f"[dim]{MEASUREMENT_UNITS[metric]}[/dim]"
            )
            raise typer.Exit(1)
        # A single value has no block around it to say who it is about, so the
        # subject travels with it — a bare number is exactly how a prompt-only
        # quantity gets mistaken for a fact about the model.
        reference = _prompt_reference_model(metric, p) if prompt_only else None
        if output_json:
            payload = {"metric": metric, "value": value,
                       "unit": MEASUREMENT_UNITS[metric], "subject": subject}
            if prompt_only:
                payload["reference_model"] = reference
                payload["about"] = "the prompt, not the model named in --model"
            print(json.dumps(payload))
        else:
            console.print(f"{metric} = {value:.6g}")
            console.print(f"[dim]{MEASUREMENT_UNITS[metric]}[/dim]")
            console.print(f"[dim]subject: {subject}[/dim]")
            if prompt_only:
                console.print(
                    f"[dim]This describes the prompt under reference model "
                    f"{reference or 'unknown'}, not {model_name}.[/dim]"
                )
        return

    if output_json:
        # Derived signals only — the raw profile (per-step distributions) is
        # never emitted here. Full artifacts are the --trace opt-in, and the
        # record links to them via trace_path.
        extras: dict = {}
        if application:
            extras["application"] = application
        if analysis_window:
            extras["analysis_window"] = analysis_window
        if input_truncated:
            extras["input_truncated"] = True
            extras["input_truncate_tokens"] = truncate
        record = _signals_record(
            p,
            model_name=model_name,
            backend=backend,
            regime=regime,
            seed=seed,
            prompt=prompt,
            latency=timings,
            trace_path=str(trace_path) if trace_path is not None else None,
            include_units=units,
            extras=extras,
        )
        print(json.dumps(record, indent=2))
        return

    console.print(f"[green]Profile complete.[/green] Hash: {h}")
    console.print()

    _print_output_text(p)

    if verbose:
        _print_verbose_io(p)

    _print_measurements(p)

    # Region sensitivity (perturbation-JSD heatmap; multimodal runs only)
    rs = getattr(p, "region_sensitivity", None)
    if rs is not None:
        console.print("[bold]Region sensitivity[/bold]")
        console.print(
            "[dim]Cells that materially affected the model's response behavior[/dim]"
        )
        console.print(rs.to_text_grid())
        console.print()

    if verbose:
        _print_verbose_stats(p)
        _print_latency(timings)

    if trace_path is not None:
        console.print(f"Trace artifact: [cyan]{trace_path}[/cyan]")
    if output_dir is not None:
        console.print(f"Outputs written to: [cyan]{output_dir}/[/cyan]")


def _print_measurements(p) -> None:
    """The measurement set, one row per quantity, in natural units.

    There is no Level column and no Normalized column, by design. A level is
    an inference requiring a null this project never established; the
    normaliser (log2 of the vocabulary size) put tokenizer metadata into a
    column labelled behaviour. Absent measurements are named and left absent.

    Quantities whose subject on this run is the prompt rather than the model
    get their own table below, for the same reason the record puts them in
    their own block: they are not measurements of this model.
    """
    vals = _measurements(p)
    subjects = _run_subjects(p)
    input_surrogate = p.findings.surrogate_model_name
    output_surrogate = p.findings.output_distribution_surrogate_name

    # No Subject column: the whole table is one subject — this model. Rows
    # that would have said something else are not in it. The one exception a
    # reader must not miss is `mixed`, which is marked.
    table = Table(title=f"Measurements — {p.model.name}", show_header=True)
    table.add_column("Quantity", style="bold", no_wrap=True)
    table.add_column("Value", justify="right")
    table.add_column("Unit / definition")

    any_mixed = False
    for m in MEASUREMENT_REGISTRY:
        subject = subjects.get(m.key)
        if subject == SUBJECT_PROMPT_ONLY:
            continue
        starred = (m.surrogate_group == "input" and input_surrogate) or (
            m.surrogate_group == "output" and output_surrogate
        )
        marks = " *" if starred else ""
        if subject == SUBJECT_MIXED:
            marks += " †"
            any_mixed = True
        v = vals.get(m.key)
        table.add_row(
            f"{m.name}{marks}",
            ABSENT_TEXT if v is None else f"{v:.6g}",
            m.unit,
        )
    console.print(table)

    surrogate_names = sorted({n for n in (input_surrogate, output_surrogate) if n})
    if surrogate_names:
        names = ", ".join(repr(n) for n in surrogate_names)
        console.print(
            f"[dim]* computed via surrogate model {names} (teacher-forcing "
            f"proxy) — a measurement of the surrogate over the target's text, "
            f"not of the target model.[/dim]"
        )
    if any_mixed:
        console.print(
            "[dim]† subject 'mixed': couples a series derived from the target "
            "with one derived from the surrogate, so it is a claim about the "
            "pair rather than about the target alone.[/dim]"
        )
    console.print(
        f"[dim]Similarity trend slope: {p.findings.similarity_trend_slope:+.6g} "
        "(per-step input/output cosine similarity, OLS slope).[/dim]"
    )
    console.print(
        "[dim]No thresholds, levels, or verdicts: this instrument describes "
        "behaviour, it does not decide anything. Run `hif schema` for the full "
        "unit definitions.[/dim]"
    )
    console.print()
    _print_prompt_measurements(p, subjects)


def _print_prompt_measurements(p, subjects: dict) -> None:
    """Prompt-only quantities, kept out of the model's measurement table.

    Printed only when the run produced any. These are real measurements — of
    the prompt, under the reference model named beside each one. They are not
    caveated facts about the model under test; nothing the model did enters
    them, so they cannot vary with its behaviour.
    """
    from hif.profile.signals import _prompt_reference_model

    vals = _prompt_measurements(p)
    if not vals:
        return

    table = Table(title="Prompt measurements — not about this model", show_header=True)
    table.add_column("Quantity", style="bold", no_wrap=True)
    table.add_column("Value", justify="right")
    table.add_column("Reference model", no_wrap=True)
    table.add_column("Unit")

    for m in MEASUREMENT_REGISTRY:
        if m.key not in vals:
            continue
        ref = _prompt_reference_model(m.key, p)
        table.add_row(m.name, f"{vals[m.key]:.6g}", ref or "unknown", m.unit)
    console.print(table)
    console.print(
        "[dim]Subject: prompt-only. Computed from the prompt text under the "
        "reference model shown, with no input from "
        f"{p.model.name} — comparable across targets for exactly that reason, "
        "and reported here rather than as a caveated measurement of the "
        "model. See docs/MEASUREMENTS.md § Subject.[/dim]"
    )
    console.print()


def _print_output_text(p) -> None:
    """Always shown (not just --verbose): the model's full generated output text,
    so a Hash/Findings result can be associated with what the model actually said."""
    output_text = "".join(s.selected_token_str for s in p.output_side.steps)
    console.print(f"[bold]Output[/bold] ({len(p.output_side.generated_ids)} tokens)")
    console.print(f"[dim]{output_text}[/dim]")
    console.print()


def _print_verbose_io(p) -> None:
    """--verbose: additionally show the model's input text and perturbation variants
    (the output text itself is always shown — see _print_output_text)."""
    console.print("[bold]Input[/bold]")
    console.print(f"[dim]{p.prompt.text}[/dim]")
    console.print()

    if p.perturbations:
        console.print("[bold]Perturbation variants[/bold]")
        for rec in p.perturbations:
            for i, variant in enumerate(rec.variants):
                js = rec.sensitivity[i].mean_js_divergence if i < len(rec.sensitivity) else None
                js_str = f"  [cyan]JSD={js:.4f}[/cyan]" if js is not None else ""
                console.print(f"  ({rec.generator} #{i + 1}){js_str}")
                console.print(f"  [dim]{variant[:200]}[/dim]")
        console.print()


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as M:SS.mmm (e.g. 0:02.417, 1:07.031)."""
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}:{rem:06.3f}"


def _print_latency(timings: dict[str, float]) -> None:
    """--verbose: wall-clock timings for each pipeline stage."""
    table = Table(title="Latency", show_header=True)
    table.add_column("Stage", style="bold")
    table.add_column("Duration", justify="right")
    for stage, secs in timings.items():
        table.add_row(stage, _fmt_duration(secs))
    console.print(table)
    console.print()


def _print_verbose_stats(p) -> None:
    """--verbose: the raw numbers behind the measurement table, plus the
    effective-config notes that change what those numbers mean."""
    table = Table(title="Stats", show_header=True)
    table.add_column("Stat", style="bold")
    table.add_column("Value", justify="right")

    st = p.metrics.stability

    def _fmt_optional(v) -> str:
        # None = absent — never a number.
        return ABSENT_TEXT if v is None else f"{v:.6g}"

    table.add_row("n_perturbation_variants", str(st.n_perturbations))
    table.add_row("input_mean_entropy (bits)", f"{p.input_side.mean_entropy:.6g}")
    table.add_row("input_mean_surprisal (bits)", f"{p.input_side.mean_surprisal:.6g}")
    table.add_row("max_entropy log2|V| (bits)", f"{p.input_side.max_entropy:.6g}")
    if p.metrics.distribution:
        ents = [d.entropy_bits for d in p.metrics.distribution]
        table.add_row("mean_output_entropy (bits)", f"{sum(ents) / len(ents):.6g}")
        table.add_row("max_output_entropy (bits)", f"{max(ents):.6g}")
        table.add_row("min_output_entropy (bits)", f"{min(ents):.6g}")
    exp = getattr(p, "exposure", None) or getattr(p, "hallucination", None)
    if exp is not None and getattr(exp, "candidates", None):
        table.add_row(
            "exposure (divergent steps / analysed)",
            f"{len(exp.high_risk_steps)}/{len(exp.candidates)}",
        )
    table.add_row("center entropy_ratio (out/in)",
                  _fmt_optional(p.center.entropy_ratio))
    table.add_row("prompt/output cosine distance",
                  f"{p.center.prompt_output_cosine_distance:.6g}")
    table.add_row("input_tokens", str(len(p.input_side.prompt_token_ids)))
    table.add_row("output_tokens", str(len(p.output_side.generated_ids)))

    # Effective-config notes: adjustments that change what the numbers mean.
    requested_k = p.config.generation.top_k
    effective_k = p.output_side.top_k
    if effective_k != requested_k:
        table.add_row("top-K", f"{effective_k} (backend max; {requested_k} requested)")
    else:
        table.add_row("top-K", str(effective_k))
    table.add_row("embedder", p.config.embedding.model_name)

    console.print(table)
    console.print()


def _live_models_for_backend(name: str) -> tuple[list[str] | None, str | None]:
    """Query a backend's actual model catalog right now.

    Returns (models, note). `models` is None when there's no live catalog to
    query for this backend (e.g. any HF repo id is eligible) or the query
    couldn't run (missing dep/credential/service) — `note` explains why.
    Static `example_models` in capabilities.py illustrate the shape of a model
    id but drift out of date as providers ship and retire models; this hits
    the provider directly so `--list` never goes stale.
    """
    import os

    if name == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None, "ANTHROPIC_API_KEY not set — showing examples instead."
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            return [m.id for m in client.models.list(limit=100)], None
        except Exception as exc:  # noqa: BLE001
            return None, f"couldn't reach Anthropic's models API ({exc})."
    if name in ("openai", "openai-vlm"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None, "OPENAI_API_KEY not set — showing examples instead."
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            return sorted(m.id for m in client.models.list()), None
        except Exception as exc:  # noqa: BLE001
            return None, f"couldn't reach OpenAI's models API ({exc})."
    if name == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None, "GEMINI_API_KEY not set (Vertex AI credentials aren't queryable this way) — showing examples instead."
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            # genai lists names as "models/gemini-2.5-flash"; strip the prefix
            # so what's printed matches what --backend gemini/config.name expects.
            return [(m.name or "").removeprefix("models/") for m in client.models.list()], None
        except Exception as exc:  # noqa: BLE001
            return None, f"couldn't reach Gemini's models API ({exc})."
    if name == "ollama":
        try:
            import httpx  # type: ignore
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            resp = httpx.get(f"{host}/api/tags", timeout=1.5)
            if resp.status_code != 200:
                return None, f"ollama server not reachable at {host} — showing examples instead."
            pulled = [m.get("name", "") for m in resp.json().get("models", [])]
            return pulled, None if pulled else "no models pulled — run `ollama pull <model>`."
        except Exception:  # noqa: BLE001
            return None, "ollama server not reachable — run `ollama serve` — showing examples instead."
    # hf / tlens / hf-vlm: any HF repo id is eligible, there's no fixed catalog.
    return None, "any Hugging Face repo id is eligible — no fixed catalog to list."


def _check_surrogate_candidates() -> list[tuple[str, str]]:
    """Check each recommended --surrogate-model candidate against the live HF Hub.

    Returns (model_id, status) pairs, status one of "ok", "gated", "not found".
    A repo can be renamed, deleted, or re-gated after the fact — this is the
    same "don't trust a static example list" check as _live_models_for_backend,
    applied to surrogate models instead of hosted-API backends.
    """
    from hif.models.capabilities import SURROGATE_CANDIDATES

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return [(m, "unknown (huggingface_hub not installed)") for m in SURROGATE_CANDIDATES]

    api = HfApi()
    results = []
    for model_id in SURROGATE_CANDIDATES:
        try:
            info = api.model_info(model_id)
            results.append((model_id, "gated" if info.gated else "ok"))
        except GatedRepoError:
            results.append((model_id, "gated"))
        except RepositoryNotFoundError:
            results.append((model_id, "not found"))
        except Exception as exc:  # noqa: BLE001
            results.append((model_id, f"error ({exc})"))
    return results


@app.command()
def models(
    backend: Optional[str] = typer.Option(
        None, help="Show only this backend (hf, tlens, ollama, openai, anthropic, gemini, hf-vlm, openai-vlm)."
    ),
    list_live: bool = typer.Option(
        False, "--list",
        help="Query each backend's actual model catalog right now (needs the provider's API key, or a "
             "running Ollama server) instead of showing static examples — use this when an example model "
             "from the docs turns out to be retired/unavailable.",
    ),
    surrogates: bool = typer.Option(
        False, "--surrogates",
        help="List recommended --surrogate-model choices (small open-weight models for recovering "
             "input-side signals on closed/Ollama backends via --surrogate) and check each is currently "
             "reachable and ungated on the Hugging Face Hub.",
    ),
) -> None:
    """List the backends you can profile, example models, and which signals each supports.

    Use this to answer "what can I test, and what will I get?" before running a
    profile. Signal availability depends on the backend: input-side signals
    (Stability, Surprise, I/O Correlation, Wager) need teacher forcing — open
    models only. The attention readings (Spread, Horizon) do NOT depend on the
    backend: they come from a separate analysis encoder reading text, so every
    backend can produce them once --diagnostics runs that stage.
    Pass --list to check live model availability instead of static examples.
    Pass --surrogates to check --surrogate-model candidates instead.
    """
    from rich.markup import escape
    from hif.models.capabilities import BACKENDS, signals_available

    if surrogates:
        console.print(
            "\n[bold]--surrogate-model candidates[/bold]  [dim](small open-weight models for "
            "--surrogate on backends that can't teacher-force)[/dim]\n"
        )
        for model_id, status in _check_surrogate_candidates():
            marker = "[green]✓ ok[/green]" if status == "ok" else f"[yellow]{status}[/yellow]"
            default_tag = "  [dim](default)[/dim]" if model_id == "unsloth/Llama-3.2-1B" else ""
            console.print(f"  {model_id:<28} {marker}{default_tag}")
        console.print()
        return

    infos = [BACKENDS[backend]] if backend and backend in BACKENDS else list(BACKENDS.values())
    if backend and backend not in BACKENDS:
        err_console.print(f"[red]Unknown backend {backend!r}. Known: {', '.join(BACKENDS)}[/red]")
        raise typer.Exit(1)

    for info in infos:
        tf = "[green]yes[/green]" if info.teacher_forcing else "[dim]no[/dim]"
        console.print(
            f"\n[bold]{info.name}[/bold]  [dim]({info.kind})[/dim]  "
            f"teacher-forcing: {tf}  ·  logprobs: {info.logprobs}"
        )
        console.print(f"  [dim]deps:[/dim]  {escape(info.deps)}")
        console.print(f"  [dim]setup:[/dim] {escape(info.setup)}")
        if list_live:
            live, note = _live_models_for_backend(info.name)
            if live is not None:
                console.print(f"  [green]models (live):[/green] {', '.join(live) if live else '[dim]none[/dim]'}")
            else:
                console.print(f"  [dim]models:[/dim] {', '.join(info.example_models)}  [yellow]({note})[/yellow]")
        else:
            console.print(f"  [dim]models:[/dim] {', '.join(info.example_models)}")
        if info.notes:
            console.print(f"  [dim]{info.notes}[/dim]")
        avail = signals_available(info.name)
        ok = [m for m, v in avail.items() if v]
        no = [m for m, v in avail.items() if not v]
        console.print(f"  [green]✓ signals:[/green] {', '.join(ok)}")
        if no:
            console.print(f"  [yellow]✗ unavailable:[/yellow] {', '.join(no)}")
        console.print(
            "  [dim]attention rows need --diagnostics, not a backend: they are "
            "an analysis encoder's attention over text, so they are available "
            "here as on every backend.[/dim]"
        )
        _print_subject_degradation(info)
    console.print()


def _print_subject_degradation(info) -> None:
    """Which measurements stop being about the target on this backend.

    Two separate statements, and the difference matters. Some quantities are
    prompt-only on every backend — no access tier can make them about a model.
    Others are the target's own when the backend teacher-forces, and become
    prompt-only when `--surrogate` reads the prompt in the target's place; on
    those backends they leave `measurements` for `prompt_measurements`.
    """
    from hif.profile.signals import SUBJECT_PROMPT_ONLY as _PO

    always = [
        m.key for m in MEASUREMENT_REGISTRY
        if m.subject == _PO and m.subject_under_surrogate is None
    ]
    if always:
        console.print(
            f"  [yellow]⊘ never about the target:[/yellow] {', '.join(always)} "
            "[dim](prompt-only on every backend)[/dim]"
        )
    if info.teacher_forcing:
        return
    degrades = [
        m.key for m in MEASUREMENT_REGISTRY
        if m.subject_under_surrogate == _PO
    ]
    if degrades:
        console.print(
            f"  [yellow]⊘ prompt-only under --surrogate:[/yellow] "
            f"{', '.join(degrades)}"
        )
        console.print(
            "  [dim]This backend cannot teacher-force, so --surrogate reads "
            "the prompt with a small local model instead. Those numbers "
            "describe the prompt under that reference model, not this "
            "backend's model, and are reported in `prompt_measurements` "
            "rather than `measurements`.[/dim]"
        )


@app.command()
def doctor() -> None:
    """Preflight check: dependencies, running services, credentials, and per-backend readiness.

    Run this first. It probes each backend's optional dependencies, checks for
    API keys / a running Ollama server, and reports which backends are ready to
    profile right now — so you don't discover a missing dep or unpulled model
    mid-pipeline.
    """
    import importlib.util
    import os

    from rich.markup import escape
    from hif.models.capabilities import BACKENDS

    def _has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    console.print("\n[bold]HIF doctor[/bold] — environment & backend readiness\n")

    # Core deps
    core_ok = _has("plotly") and _has("numpy")
    console.print(f"  core (numpy, plotly): {'[green]ok[/green]' if core_ok else '[red]missing[/red]'}")
    console.print(f"  embedder (sentence-transformers): "
                  f"{'[green]ok[/green]' if _has('sentence_transformers') else '[yellow]missing — run: pip install sentence-transformers[/yellow]'}")

    # Ollama server reachability
    ollama_up = False
    try:
        import httpx  # type: ignore
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        pulled = []
        try:
            resp = httpx.get(f"{host}/api/tags", timeout=1.5)
            ollama_up = resp.status_code == 200
            if ollama_up:
                pulled = [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:  # noqa: BLE001
            ollama_up = False
        if ollama_up:
            plist = ", ".join(pulled) if pulled else "[dim]none pulled — run `ollama pull llama3.2`[/dim]"
            console.print(f"  ollama server ({host}): [green]running[/green] · pulled: {plist}")
        else:
            console.print(f"  ollama server ({host}): [yellow]not reachable — run `ollama serve`[/yellow]")
    except Exception:  # noqa: BLE001
        console.print("  ollama client (httpx): [dim]not installed — pip install 'hif[ollama]'[/dim]")

    # Credentials
    console.print("\n[bold]credentials[/bold]")
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "HF_TOKEN"):
        console.print(f"  {env}: {'[green]set[/green]' if os.environ.get(env) else '[dim]unset[/dim]'}")

    # Per-backend readiness
    console.print("\n[bold]backends[/bold]")
    dep_probe = {
        "hf": "transformers", "tlens": "transformer_lens", "hf-vlm": "transformers",
        "ollama": "httpx", "openai": "openai", "anthropic": "anthropic",
        "gemini": "google.genai", "openai-vlm": "openai",
    }
    cred_probe = {
        "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY", "openai-vlm": "OPENAI_API_KEY",
    }
    for name, info in BACKENDS.items():
        dep = dep_probe.get(name, "")
        dep_ok = _has(dep.split(".")[0]) if dep else True
        issues = []
        if not dep_ok:
            issues.append(f"missing dep ({escape(info.deps)})")
        cred = cred_probe.get(name)
        if cred and not os.environ.get(cred):
            issues.append(f"{cred} unset")
        if name == "ollama" and not ollama_up:
            issues.append("server not running")
        status = "[green]ready[/green]" if not issues else f"[yellow]{'; '.join(issues)}[/yellow]"
        console.print(f"  {name:11s} {status}")

    console.print("\n[dim]The full measurement set needs an open-weight backend: "
                  "`hif profile gpt2 \"hello\" --backend hf`. "
                  "Run `hif models` for the full capability matrix.[/dim]\n")


@app.command()
def suite(
    model_name: str = typer.Argument(..., help="Model name (e.g. gpt2)"),
    regime: Optional[str] = typer.Option(None, help="Single regime; None = all eight"),
    backend: str = typer.Option("hf", help="Model backend: hf | tlens | ollama"),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Path = typer.Option(Path("outputs"), help="Output directory"),
    max_new_tokens: int = typer.Option(64, help="Maximum new tokens to generate"),
    top_k: int = typer.Option(50, help="Top-K candidates per step"),
    charts: bool = typer.Option(
        False,
        "--charts",
        help="Generate plots + the combined dashboard locally (off by default).",
    ),
    units: bool = typer.Option(
        False, "--units",
        help="Include a per-measurement units block in each record. Constant per "
             "signal_set_version and identical on every record, so off by default; "
             "`hif schema` prints the same information without running a model.",
    ),
) -> None:
    """Run the full HI pipeline over the prompt suite (every regime, or one)."""
    from hif.prompts.suite import REGIMES, get_regime

    if regime is not None:
        try:
            selected_regimes = [get_regime(regime)]
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    else:
        selected_regimes = list(REGIMES)

    console.print(f"[bold]HI Suite[/bold]")
    console.print(f"  Model:   {model_name} ({backend})")
    console.print(f"  Regimes: {len(selected_regimes)}")
    console.print(f"  Seed:    {seed}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Loading model...", total=None)
        model = _load_model(model_name, backend)
        progress.update(t, description="Loading embedder...")
        embedder = _load_embedder()

    n_ok = 0
    n_failed = 0

    for reg in selected_regimes:
        console.print(f"[bold]Regime:[/bold] {reg.name}")
        for prompt_text in reg.prompts:
            console.print(f"  [dim]{prompt_text[:70]!r}[/dim]")
            try:
                p, _ = _run_single_profile(
                    model_name=model_name,
                    prompt=prompt_text,
                    regime=reg.name,
                    backend=backend,
                    seed=seed,
                    output_dir=output_dir / reg.name,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    charts=charts,
                    model=model,
                    embedder=embedder,
                )
            except Exception as exc:  # noqa: BLE001 — per-prompt isolation
                console.print(f"    [red]error: {exc}[/red]")
                _emit_json_line({
                    "schema_version": SIGNAL_RECORD_VERSION,
                    "regime": reg.name,
                    "prompt": prompt_text,
                    "error": str(exc) or exc.__class__.__name__,
                })
                n_failed += 1
                continue
            _emit_json_line(_signals_record(
                p,
                model_name=model_name,
                backend=backend,
                regime=reg.name,
                seed=seed,
                prompt=prompt_text,
                include_units=units,
            ))
            n_ok += 1

    console.print(
        f"suite complete: {n_ok} ok, {n_failed} failed. "
        f"Reports written to {output_dir}/"
    )
    if n_ok == 0:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# hif compare
# ---------------------------------------------------------------------------
#
# Compare reports the difference between two profiles, measurement by
# measurement, in natural units. It deliberately does NOT emit a verdict.
#
# The previous version bucketed each delta as consistent/moderate/high against
# fixed thresholds and printed "SIGNATURE HELD / SHIFTED / MOVED", exiting 2 on
# MOVED so it could be wired into a release gate. That decision rule was
# audited against pairs of runs known to be identical and produced a ~43%
# false-positive rate. Thresholds and decisions are out of scope for this
# instrument; a delta and its unit are what it can honestly report.


@app.command()
def compare(
    profile_a: Path = typer.Argument(..., help="Path to the first profile JSON"),
    profile_b: Path = typer.Argument(..., help="Path to the second profile JSON"),
    output: Optional[Path] = typer.Option(None, help="Optional output Markdown file"),
    output_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Report the per-measurement difference between two profiles.

    Every row is (A, B, B − A) in the measurement's own natural unit. There is
    no verdict, no band, and no non-zero exit on divergence: deciding whether a
    difference matters needs a null distribution for your workload, which this
    tool does not have and does not fabricate.

    Exit codes: 0 normally, 1 for an unreadable file, 2 for a comparison that
    cannot be made at all (different modalities, or no shared measurements).
    """
    from hif.profile.schema import BehavioralRangeProfile

    for path in (profile_a, profile_b):
        if not path.exists():
            err_console.print(f"[red]File not found: {path}[/red]")
            raise typer.Exit(1)

    raw_a = json.loads(profile_a.read_text())
    raw_b = json.loads(profile_b.read_text())
    pa = BehavioralRangeProfile.model_validate(raw_a)
    pb = BehavioralRangeProfile.model_validate(raw_b)

    # Cross-modality comparison is a different experimental condition — hard
    # error, never a warning. Missing modality on an older profile reads "text".
    modality_a = _profile_modality(pa)
    modality_b = _profile_modality(pb)
    if modality_a != modality_b:
        _modality_mismatch_exit(modality_a, modality_b)

    version_a = _artifact_signal_set_version(raw_a)
    version_b = _artifact_signal_set_version(raw_b)
    if _signal_set_family(version_a) != _signal_set_family(version_b):
        _signal_set_mismatch_exit(version_a, version_b)

    vals_a = _measurements(pa)
    vals_b = _measurements(pb)
    shared = [k for k in MEASUREMENT_KEYS if k in vals_a and k in vals_b]
    if not shared:
        err_console.print(
            "[red]These profiles share no measurements — the comparison would "
            "be empty.[/red]"
        )
        raise typer.Exit(2)

    only_a = [k for k in MEASUREMENT_KEYS if k in vals_a and k not in vals_b]
    only_b = [k for k in MEASUREMENT_KEYS if k in vals_b and k not in vals_a]
    for key in only_a:
        err_console.print(
            f"[yellow]excluded: {key} — absent from {profile_b.name}[/yellow]"
        )
    for key in only_b:
        err_console.print(
            f"[yellow]excluded: {key} — absent from {profile_a.name}[/yellow]"
        )

    if output_json:
        result = {
            "profile_a": str(profile_a),
            "profile_b": str(profile_b),
            "model_a": pa.model.name,
            "model_b": pb.model.name,
            "measurements_a": vals_a,
            "measurements_b": vals_b,
            "delta": {k: vals_b[k] - vals_a[k] for k in shared},
            "units": {k: MEASUREMENT_UNITS[k] for k in shared},
            "excluded": {"only_in_a": only_a, "only_in_b": only_b},
        }
        print(json.dumps(result, indent=2))
        return

    table = Table(title="Profile comparison", show_header=True)
    table.add_column("Measurement", style="bold", no_wrap=True)
    table.add_column(pa.model.name, justify="right")
    table.add_column(pb.model.name, justify="right")
    table.add_column("B − A", justify="right")
    table.add_column("Unit")
    for key in shared:
        table.add_row(
            key,
            f"{vals_a[key]:.6g}",
            f"{vals_b[key]:.6g}",
            f"{vals_b[key] - vals_a[key]:+.6g}",
            MEASUREMENT_UNITS[key].split(" — ")[0],
        )
    console.print(table)
    console.print(
        "[dim]Differences only. Whether a difference matters depends on a null "
        "distribution for your workload, which this tool does not have.[/dim]"
    )

    if output is not None:
        lines = [
            "# hif profile comparison",
            "",
            f"- A: `{profile_a}` ({pa.model.name})",
            f"- B: `{profile_b}` ({pb.model.name})",
            "",
            "| Measurement | A | B | B − A | Unit |",
            "|---|---|---|---|---|",
        ]
        for key in shared:
            lines.append(
                f"| {key} | {vals_a[key]:.6g} | {vals_b[key]:.6g} | "
                f"{vals_b[key] - vals_a[key]:+.6g} | "
                f"{MEASUREMENT_UNITS[key].split(' — ')[0]} |"
            )
        lines += [
            "",
            "Differences only. Whether a difference matters depends on a null "
            "distribution for your workload, which this tool does not have.",
            "",
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines))
        console.print(f"\nComparison written to: [cyan]{output}[/cyan]")


def _resolve_validation_corpus(corpus: Optional[Path], seed: int, quiet: bool) -> Path:
    """Return a corpus directory, generating the built-in known-answer corpus
    into ~/.hif/validation-corpus/<seed>/ on first use (deterministic from
    the seed; images are not shipped in the package)."""
    if corpus is not None:
        return corpus
    from hif.validation.corpus import generate_corpus

    cache_dir = Path.home() / ".hif" / "validation-corpus" / str(seed)
    if not (cache_dir / "corpus.jsonl").exists():
        if not quiet:
            console.print(f"[dim]Generating validation corpus into {cache_dir}...[/dim]")
        generate_corpus(seed=seed, out_dir=cache_dir)
    return cache_dir


@app.command("validate-model")
def validate_model(
    model_name: str = typer.Argument(..., help="Model name (e.g. a HF VLM checkpoint or gpt-4o)"),
    backend: str = typer.Option(..., "--backend", help="Model backend: hf-vlm | openai-vlm"),
    grid: Optional[str] = typer.Option(
        None, "--grid",
        help="Mask grid as ROWSxCOLS (default: 4x4; 2x2 with --pilot).",
    ),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus",
        help="Directory containing a corpus.jsonl known-answer suite "
        "(default: built-in suite, generated to ~/.hif/validation-corpus/ on first use).",
    ),
    pilot: bool = typer.Option(
        False, "--pilot",
        help="Fast smoke run: 4 images on a 2x2 grid instead of 10 images on 4x4.",
    ),
    seed: int = typer.Option(20260703, help="Corpus-generation / run seed"),
    yes: bool = typer.Option(False, "--yes", help="Skip the full-run confirmation prompt"),
    output_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress output"),
) -> None:
    """Validate region-sensitivity measurement for a model against HIF's known-answer suite.

    Runs the model on synthetic tasks where the task-relevant region of each
    image is known by construction, and checks that region sensitivity locates
    it. PASS means the instrument is validated against ground-truth synthetic
    tasks for this model — it is not an accuracy claim about your workload.

    Exit codes: 0 pass, 2 fail, 3 usage error.
    """
    if backend not in ("hf-vlm", "openai-vlm"):
        err_console.print("[red]--backend must be 'hf-vlm' or 'openai-vlm'[/red]")
        raise typer.Exit(3)


    if grid is None:
        grid_tuple = (2, 2) if pilot else (4, 4)
    else:
        try:
            r, c = grid.lower().split("x")
            grid_tuple = (int(r), int(c))
            if grid_tuple[0] <= 0 or grid_tuple[1] <= 0:
                raise ValueError
        except ValueError:
            err_console.print(f"[red]--grid must look like 4x4, got {grid!r}[/red]")
            raise typer.Exit(3)

    corpus_dir = _resolve_validation_corpus(corpus, seed, quiet or output_json)

    if not pilot and not yes:
        n_cells = grid_tuple[0] * grid_tuple[1]
        console.print(
            f"[yellow]Full validation runs 10 images x 3 question variants x "
            f"({n_cells} masked cells + 1 baseline) = {10 * 3 * (n_cells + 1)} "
            "inference runs. This can take a while on local models and incurs "
            "API cost on hosted backends. Use --pilot for a fast smoke run.[/yellow]"
        )
        if not typer.confirm("Proceed?"):
            raise typer.Exit(3)

    from hif.validation.harness import validate_region_sensitivity

    if not quiet and not output_json:
        console.print(
            f"[dim]Validating {model_name} ({backend}) on "
            f"{grid_tuple[0]}x{grid_tuple[1]} grid"
            f"{' (pilot subset)' if pilot else ''}...[/dim]"
        )

    try:
        model = _load_model(model_name, backend)
        result = validate_region_sensitivity(
            model, corpus_dir=corpus_dir, grid=grid_tuple, pilot=pilot, seed=seed,
        )
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3)
    except typer.Exit:
        raise
    except Exception as exc:
        err_console.print(f"[red]Validation run failed: {exc}[/red]")
        raise typer.Exit(1)

    if output_json:
        payload = {
            "model": result.model_id,
            "backend": backend,
            "grid": f"{result.grid[0]}x{result.grid[1]}",
            "threshold": result.threshold,
            "top2_rate": round(result.top2_rate, 4),
            "passed": result.passed,
            "per_image": [
                {
                    "image_id": rec.image_id,
                    "variant": rec.variant,
                    "answer_cell": rec.answer_cell,
                    "answer_cell_rank": rec.answer_cell_rank,
                    "in_top2": rec.in_top2,
                    "cell_jsd": {k: round(v, 6) for k, v in rec.cell_jsd.items()},
                }
                for rec in result.per_image
            ],
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        table = Table(
            title=f"Region-sensitivity validation — {result.model_id}",
            show_header=True,
        )
        table.add_column("Image", style="bold")
        table.add_column("Variant")
        table.add_column("Answer cell")
        table.add_column("Rank", justify="right")
        table.add_column("Top-2")
        for rec in result.per_image:
            table.add_row(
                rec.image_id,
                str(rec.variant),
                f"({rec.answer_cell['row']},{rec.answer_cell['col']})",
                str(rec.answer_cell_rank),
                "[green]yes[/green]" if rec.in_top2 else "[red]no[/red]",
            )
        console.print(table)
        console.print()
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(
            f"{status} — answer cell in top 2 for {result.top2_rate:.0%} of "
            f"runs (threshold {result.threshold:.0%}, "
            f"grid {result.grid[0]}x{result.grid[1]}, n={len(result.per_image)})."
        )
        console.print(
            "[dim]Validated against ground-truth synthetic tasks — a statement "
            "about the measurement instrument on this model, not about your "
            "workload.[/dim]"
        )

    if not result.passed:
        raise typer.Exit(2)


@app.command()
def render(
    profile_json: Path = typer.Argument(..., help="Path to profile JSON"),
    public: bool = typer.Option(False, help="Produce public-facing summary instead of technical"),
    output: Optional[Path] = typer.Option(None, help="Output path (default: alongside JSON)"),
) -> None:
    """Load an existing profile from JSON and re-render Markdown."""
    from hif.profile.render_markdown import render_public, render_technical
    from hif.profile.schema import BehavioralRangeProfile

    if not profile_json.exists():
        console.print(f"[red]File not found: {profile_json}[/red]")
        raise typer.Exit(1)

    p = BehavioralRangeProfile.model_validate_json(profile_json.read_text())

    suffix = "_public.md" if public else "_technical.md"
    if output is None:
        output = profile_json.with_suffix("").with_name(
            profile_json.stem + suffix
        )

    if public:
        render_public(p, output)
    else:
        render_technical(p, output)

    console.print(f"[green]Rendered to:[/green] {output}")


@app.command()
def schema(
    output_json: bool = typer.Option(
        True, "--json/--text",
        help="Emit the machine-readable schema document (default) or a human table.",
    ),
) -> None:
    """Print the measurement registry: every key with its full row.

    Each measurement is emitted as its complete registry row — key, name,
    label, unit, definition, and its triple (observable, functional,
    resolution). This is the contract for `hif profile --json`, `hif suite`
    and `hif batch` records, and the machine-readable mirror of
    docs/MEASUREMENTS.md. Every measurement is in natural units; there is no
    normalised variant and no level.
    """
    if output_json:
        print(json.dumps({
            "schema_version": SIGNAL_RECORD_VERSION,
            "signal_set_version": SIGNAL_SET_VERSION,
            "stdout_format": {
                "hif profile --json": "a single JSON document",
                "prompt_measurements": "present only when the run produced "
                        "prompt-only quantities (see \"subjects\"); they are "
                        "never inside \"measurements\", which carries "
                        "measurements of the model named in the record and "
                        "nothing else.",
                "hif suite": "JSONL, one record per prompt",
                "hif batch": "JSONL, one record per workload row",
                "hif compare --json": "a single JSON document",
                "note": "stdout carries JSON only; progress, warnings and errors "
                        "go to stderr. A failed row is still a record, carrying "
                        "an \"error\" key instead of \"measurements\".",
            },
            # What the resolution field means: the granularity of the
            # underlying series the run-level scalar summarises. Rows with a
            # per-step / per-position resolution have a token-level trace
            # behind the number; "aggregate" rows exist only at run level.
            "resolutions": {
                "aggregate": "the quantity exists only at whole-run level "
                             "(across perturbation variants, trajectory "
                             "branches, or the run's endpoints)",
                "per-step": "one sample per generation step; the scalar "
                            "summarises a per-step trace",
                "per-position": "one sample per prompt/context position; the "
                                "scalar summarises a per-position trace",
            },
            # Whose behaviour each measurement describes. `subject` is the
            # answer when the target's own machinery produced the quantity;
            # `subject_under_surrogate`, when present, is what it becomes once
            # the surrogate named by `surrogate_group` stands in.
            "subjects": dict(SUBJECT_LEGEND),
            "measurements": {
                m.key: {
                    "name": m.name,
                    "label": m.label,
                    "unit": m.unit,
                    "definition": m.definition,
                    "observable": m.observable,
                    "functional": m.functional,
                    "resolution": m.resolution,
                    "subject": m.subject,
                    "subject_under_surrogate": m.subject_under_surrogate,
                    "surrogate_group": m.surrogate_group or None,
                }
                for m in MEASUREMENT_REGISTRY
            },
        }, indent=2))
        return

    table = Table(title="hif measurement set", show_header=True)
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Label")
    table.add_column("Unit")
    table.add_column("Resolution")
    table.add_column("Subject")
    table.add_column("Definition")
    for m in MEASUREMENT_REGISTRY:
        subject = m.subject
        if m.subject_under_surrogate is not None:
            subject = f"{m.subject} → {m.subject_under_surrogate} (surrogate)"
        table.add_row(
            m.key, m.label or "—", m.unit, m.resolution, subject, m.definition
        )
    console.print(table)
    console.print("\n[bold]Subject[/bold] — whose behaviour the number describes:")
    for value, gloss in SUBJECT_LEGEND.items():
        console.print(f"  [bold]{value}[/bold] — {gloss}")


# ---------------------------------------------------------------------------
# Commands — Batch (workload runner)
# ---------------------------------------------------------------------------


def _batch_explicit_params(ctx: typer.Context) -> frozenset:
    """Which generation knobs the user passed explicitly (vs. defaults) —
    same override rule as `profile`: explicit CLI flags beat --config-file."""
    # Compare by enum NAME, not identity/equality: typer >=0.26 returns its
    # own ParameterSource enum class rather than click's, so a cross-class
    # `!=` against click.core.ParameterSource.DEFAULT is always True.
    return frozenset(
        name for name in ("max_new_tokens", "top_k", "seed", "mode")
        if (src := ctx.get_parameter_source(name)) is not None
        and getattr(src, "name", None) != "DEFAULT"
    )


def _open_records_file(output_dir: Optional[Path]):
    """Open <output-dir>/records.jsonl for the batch stream mirror (or None)."""
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir / "records.jsonl").open("w")


@app.command()
def batch(
    ctx: typer.Context,
    workload: Path = typer.Argument(
        ..., help="Workload JSONL file: one {\"query_id\", \"text\"[, \"image\", \"regime\"]} row per line."
    ),
    model_name: str = typer.Argument(..., help="Model name (e.g. gpt2)"),
    backend: str = typer.Option(
        "hf",
        help="Model backend: hf | tlens | ollama | openai | anthropic | gemini | "
        "hf-vlm | openai-vlm. Workloads containing image rows require hf-vlm "
        "or openai-vlm.",
    ),
    regime: str = typer.Option(
        "batch", help="Default prompt regime (a per-row \"regime\" key overrides it)."
    ),
    seed: int = typer.Option(42, help="Random seed"),
    max_new_tokens: int = typer.Option(64, help="Maximum new tokens to generate"),
    top_k: int = typer.Option(50, help="Top-K candidates per step"),
    config_file: Optional[Path] = typer.Option(
        None,
        help="TOML run config (tables mirror RunConfig). CLI flags you pass "
        "explicitly override the file.",
    ),
    mode: str = typer.Option(
        "fast",
        help="fast: fewer perturbation variants. audit: full perturbation set.",
    ),
    surrogate: bool = typer.Option(
        False,
        "--surrogate",
        help="Recover input-side signals on backends that cannot teacher-force "
        "by teacher-forcing a small local proxy model (see `hif profile "
        "--surrogate`). Implied by --surrogate-model.",
    ),
    surrogate_model: Optional[str] = typer.Option(
        None,
        "--surrogate-model",
        help="Open-weight HF model id used for --surrogate (default: Llama 3.2 "
        "1B, ungated mirror). Passing this implies --surrogate.",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="Opt-in traceability: persist each row's full profile artifact "
        "(raw per-step top-K distributions). Default off: compute-and-discard.",
    ),
    trace_dir: Optional[Path] = typer.Option(
        None,
        "--trace-dir",
        help="Where --trace artifacts are written (default: <output-dir>/traces, "
        "or ./traces when no --output-dir). Passing this implies --trace.",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Profile only the first N workload rows."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Also mirror the stdout record stream to <output-dir>/records.jsonl. "
        "Default: records stream to stdout only.",
    ),
    units: bool = typer.Option(
        False, "--units",
        help="Include a per-measurement units block in each record. Constant per "
             "signal_set_version and identical on every record, so off by default; "
             "`hif schema` prints the same information without running a model.",
    ),
) -> None:
    """Profile every prompt in a workload file (model loaded once).

    Streams one compact JSON record per row to stdout (pipe-safe; all
    progress/logs go to stderr). Row failures emit an error record and the
    run continues.
    """
    from hif import batch as batch_mod


    # Backend validation FIRST — cheap, and an unknown backend should fail
    # fast (exit 3) before any model load.
    # Ollama-style names ("gemma3:4b-it-qat") contain a colon, invalid in HF
    # repo ids — same auto-route as `hif profile`.
    if backend == "hf" and ":" in model_name:
        err_console.print(
            f"[yellow]{model_name!r} looks like an Ollama model tag — using "
            "--backend ollama. Pass --backend explicitly to override.[/yellow]"
        )
        backend = "ollama"
    from hif.models.factory import KNOWN_BACKENDS
    if backend not in KNOWN_BACKENDS:
        err_console.print(
            f"[red]Unknown --backend {backend!r}. "
            f"Use one of: {', '.join(KNOWN_BACKENDS)}.[/red]"
        )
        raise typer.Exit(3)

    if mode not in ("fast", "audit"):
        err_console.print(f"[red]--mode must be 'fast' or 'audit', got {mode!r}[/red]")
        raise typer.Exit(3)

    # --surrogate-model implies --surrogate; --trace-dir implies --trace
    # (same conventions as `profile`).
    if surrogate_model is not None:
        surrogate = True
    if trace_dir is not None:
        trace = True

    # Validate the whole workload up front — before any model load.
    try:
        rows = batch_mod.load_workload(workload, limit=limit)
    except batch_mod.WorkloadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3)

    if not rows:
        err_console.print(
            f"[red]Workload {workload} has no rows to profile"
            f"{' after --limit' if limit is not None else ''} — nothing to do.[/red]"
        )
        raise typer.Exit(3)

    if batch_mod.has_image_rows(rows) and backend not in batch_mod.VLM_BACKENDS:
        err_console.print(
            "[red]This workload contains image rows — use --backend hf-vlm "
            "(local VLM) or --backend openai-vlm (hosted vision API).[/red]"
        )
        raise typer.Exit(3)

    explicit = _batch_explicit_params(ctx)
    base_config = _load_config_file(config_file) if config_file is not None else None
    config = _make_run_config(
        model_name, backend, max_new_tokens, top_k, seed, output_dir,
        base=base_config, explicit=explicit,
    )
    if base_config is None or "mode" in explicit:
        config.perturbation.n_variants = 2 if mode == "fast" else 5
    config.traceability.enabled = trace
    # A --config-file [generation] seed wins over the CLI *default* (an
    # explicit --seed still beats the file) — the seed passed to the run must
    # match config.generation.seed or the record lies about reproducibility.
    if "seed" not in explicit:
        seed = config.generation.seed

    resolved_trace_dir = trace_dir or (
        (output_dir / "traces") if output_dir else Path("traces")
    )

    records_fh = _open_records_file(output_dir)

    def _emit(record: dict) -> None:
        _emit_json_line(record)
        if records_fh is not None:
            records_fh.write(json.dumps(record) + "\n")
            records_fh.flush()

    def _log(msg: str) -> None:
        err_console.print(msg, markup=False, highlight=False)

    try:
        n_ok, n_failed = batch_mod.run_batch(
            config,
            rows,
            default_regime=regime,
            seed=seed,
            surrogate=surrogate,
            surrogate_model_id=surrogate_model,
            trace=trace,
            trace_dir=resolved_trace_dir,
            emit=_emit,
            include_units=units,
            log=_log,
        )
    finally:
        if records_fh is not None:
            records_fh.close()

    _log(f"batch complete: {n_ok} ok, {n_failed} failed of {len(rows)} row(s)")
    if rows and n_ok == 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
