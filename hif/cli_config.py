"""Turning CLI flags and a --config-file into the RunConfig a run executes.

This is the precedence layer, and it is the subtlest part of the CLI: a TOML
file supplies the baseline, the model identity always comes from the CLI
arguments, and any generation knob the user typed explicitly beats the file.
Getting that order wrong changes what the numbers mean without changing what
they are called, so it lives in one place with its reasoning attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from hif.cli_base import err_console

if TYPE_CHECKING:
    from hif.config import RunConfig


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
    # Validation aliases (e.g. the pre-rename [hallucination] table for
    # [exposure]) are accepted: pydantic honours them, so this guard must too.
    from pydantic import AliasChoices

    valid_keys: set[str] = set(RunConfig.model_fields)
    for _field in RunConfig.model_fields.values():
        if isinstance(_field.validation_alias, AliasChoices):
            valid_keys.update(
                a for a in _field.validation_alias.choices if isinstance(a, str)
            )
    unknown = sorted(set(data) - valid_keys)
    if unknown:
        err_console.print(
            f"[red]Unknown key(s) in --config-file {path}: "
            f"{', '.join(unknown)}. "
            f"Valid tables: {', '.join(sorted(RunConfig.model_fields))}.[/red]"
        )
        raise typer.Exit(3)
    # Same guard one level down. RunConfig's sub-models also tolerate unknown
    # fields (embedded profile JSON must keep loading across versions), so a
    # typo'd key INSIDE a valid table ([perturbation] generatorz = ...) would
    # validate cleanly and the run would silently measure with the default —
    # the worst outcome a config file can produce, because the record carries
    # no evidence of it. The TOML path is authored by hand and gets no such
    # tolerance: every key must name a real field (or one of its aliases).
    # Fields typed as plain dicts (e.g. [model] extra_body, passed through to
    # the provider verbatim) are exempt — their keys are the provider's
    # vocabulary, not ours.
    for table_name, table_value in data.items():
        if not isinstance(table_value, dict):
            continue
        field = RunConfig.model_fields.get(table_name)
        if field is None:  # matched via alias — resolve to the real field
            field = next(
                f for f in RunConfig.model_fields.values()
                if isinstance(f.validation_alias, AliasChoices)
                and table_name in f.validation_alias.choices
            )
        sub_model = field.annotation
        sub_fields = getattr(sub_model, "model_fields", None)
        if sub_fields is None:
            continue
        valid_inner: set[str] = set(sub_fields)
        for _inner in sub_fields.values():
            if isinstance(_inner.validation_alias, AliasChoices):
                valid_inner.update(
                    a for a in _inner.validation_alias.choices if isinstance(a, str)
                )
        unknown_inner = sorted(set(table_value) - valid_inner)
        if unknown_inner:
            # \[ — rich would otherwise read [perturbation] as a style tag
            # and silently drop the table name from its own error message.
            err_console.print(
                f"[red]Unknown key(s) in \\[{table_name}] of --config-file "
                f"{path}: {', '.join(unknown_inner)}. "
                f"Valid keys: {', '.join(sorted(sub_fields))}.[/red]"
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


def _explicit_generation_params(ctx: typer.Context) -> frozenset:
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


def _check_mode(mode: str) -> None:
    """Reject a --mode value neither command accepts."""
    if mode not in ("fast", "audit"):
        err_console.print(f"[red]--mode must be 'fast' or 'audit', got {mode!r}[/red]")
        raise typer.Exit(3)


def _check_acquisition(acquisition: str) -> None:
    """Reject an --acquisition tier no command accepts.

    Shared by `profile`, `batch`, and `suite`: a ceiling that means one thing
    at one scale and something else at another would be worse than no ceiling.
    """
    from hif.profile.registry import ACQUISITIONS

    if acquisition not in ACQUISITIONS:
        err_console.print(
            f"[red]--acquisition must be one of: {', '.join(ACQUISITIONS)} — "
            f"got {acquisition!r}[/red]"
        )
        raise typer.Exit(3)
