"""`hif config` — inspect and author run configuration without running."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from hif.cli._app import (
    app,
    console,
    err_console,
)
from hif.cli._config import (
    _check_acquisition,
    _check_mode,
    _explicit_generation_params,
    _load_config_file,
)
from hif.cli._run import _resolve_run_config




config_app = typer.Typer(
    help="Inspect and author run configuration without running a profile.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


def _toml_scalar(v) -> str:
    """One TOML value. JSON string escaping is a valid TOML basic string."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_scalar(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"cannot serialise {type(v).__name__} to TOML")


def _emit_toml(data: dict, *, only_keys: dict | None = None) -> str:
    """Render a resolved config dict as a run.toml.

    The output is valid --config-file input, so `hif config show > run.toml`
    round-trips (model name/backend still come from the CLI arguments at run
    time — the [model] table's name/backend lines are informational). Keys
    whose value is None are emitted as comments: TOML has no null, and a
    silently missing line would be indistinguishable from a forgotten one.

    `only_keys` (from --diff) limits each table to the listed keys.
    """
    lines: list[str] = []
    for table, fields in data.items():
        if not isinstance(fields, dict):
            continue
        keys = fields if only_keys is None else {
            k: v for k, v in fields.items() if k in only_keys.get(table, set())
        }
        if not keys:
            continue
        lines.append(f"[{table}]")
        for key, value in keys.items():
            if value is None:
                lines.append(f"# {key} = (unset)")
            else:
                lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    return "\n".join(lines)


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    # `\[` is Rich's markup escape — a bare [model] is swallowed as a style
    # tag. tools/gen_flags_doc.py drops the backslash for docs/FLAGS.md.
    model_name: str = typer.Argument(
        "gpt2", help="Model name (affects \\[model] only)"
    ),
    backend: str = typer.Option("hf", help="Model backend"),
    config_file: Optional[Path] = typer.Option(
        None, help="TOML run config to resolve (same file `hif profile` takes)."
    ),
    mode: str = typer.Option("fast", help="fast | audit (perturbation budget)"),
    lite: bool = typer.Option(False, "--lite", help="Apply the --lite stage budget"),
    acquisition: str = typer.Option(
        "elicited-output", "--acquisition",
        help="observational | synthesized-input | elicited-output",
    ),
    max_new_tokens: int = typer.Option(64, help="Maximum new tokens"),
    top_k: int = typer.Option(50, help="Top-K candidates per step"),
    seed: int = typer.Option(42, help="Random seed"),
    diagnostics: bool = typer.Option(False, "--diagnostics", help="Apply --diagnostics"),
    diff: bool = typer.Option(
        False, "--diff",
        help="Show only departures from the schema defaults.",
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit JSON instead of TOML."
    ),
) -> None:
    """Print the fully resolved run config — without loading a model or running.

    Resolution goes through the SAME path `hif profile` uses
    (_resolve_run_config), so what this prints is what that runs; the two
    cannot drift. This is the confirmation step for any config change: author
    the file, `hif config show --config-file run.toml --diff`, and check that
    every key you set appears — a typo'd key now exits 3 at load, and a key
    you expected to change but don't see here did not apply.

    The TOML output is valid --config-file input, so `... > run.toml`
    round-trips. Secrets are redacted.
    """
    from hif.config import RunConfig, public_config_dict

    _check_mode(mode)
    _check_acquisition(acquisition)

    base_config = _load_config_file(config_file) if config_file is not None else None
    explicit = _explicit_generation_params(ctx)
    n_variants = 2 if mode == "fast" else 5

    config = _resolve_run_config(
        model_name, backend, max_new_tokens, top_k, seed, None,
        diagnostics=diagnostics, base_config=base_config, explicit=explicit,
        n_perturbation_variants=n_variants, lite=lite, acquisition=acquisition,
    )
    resolved = public_config_dict(config)

    only_keys: dict | None = None
    if diff:
        defaults = public_config_dict(RunConfig())
        only_keys = {}
        for table, fields in resolved.items():
            if not isinstance(fields, dict):
                continue
            changed = {
                k for k, v in fields.items() if defaults.get(table, {}).get(k) != v
            }
            if changed:
                only_keys[table] = changed

    if output_json:
        if only_keys is not None:
            resolved = {
                t: {k: v for k, v in f.items() if k in only_keys.get(t, set())}
                for t, f in resolved.items()
                if isinstance(f, dict) and only_keys.get(t)
            }
        print(json.dumps(resolved, indent=2))
        return

    header = (
        "# resolved run config"
        + (f" — departures from defaults only" if diff else "")
        + f"\n# {model_name} ({backend}) · mode={mode}"
        + (f" · --lite" if lite else "")
        + (f" · --acquisition {acquisition}" if acquisition != "elicited-output" else "")
        + (f" · --config-file {config_file}" if config_file else "")
    )
    print(header + "\n")
    body = _emit_toml(resolved, only_keys=only_keys)
    print(body if body.strip() else "# (no departures from defaults)")


@config_app.command("init")
def config_init(
    output: Path = typer.Option(
        Path("run.toml"), "--output", "-o", help="Where to write the template."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
) -> None:
    """Write a run.toml of pure schema defaults to edit from.

    Authoring from a canonical template beats composing tables from memory:
    every key in the file is spelled correctly and set to its default, so the
    diff between this file and yours IS your experimental condition. Keys are
    the same names `hif config show` prints and docs/CONFIG.md documents.
    """
    from hif.config import RunConfig, public_config_dict

    if output.exists() and not force:
        err_console.print(
            f"[red]{output} exists — pass --force to overwrite.[/red]"
        )
        raise typer.Exit(3)
    defaults = public_config_dict(RunConfig())
    # An EXPLICIT [generation] temperature is mirrored onto model.temperature
    # (see _make_run_config), and an unset one keeps each backend's own
    # default (0 for OpenAI). Writing the 1.0 default into the template would
    # make it explicit — silently changing API-backend sampling for everyone
    # who edits from this file. Leave it commented instead.
    generation = dict(defaults["generation"])
    default_temperature = generation.pop("temperature")
    defaults["generation"] = generation
    body = _emit_toml(defaults).replace(
        "[generation]",
        "[generation]\n"
        f"# temperature = {default_temperature}  "
        "# unset = each backend's own default (0 for OpenAI); "
        "setting it forces this value on every backend",
    )
    output.write_text(
        "# run.toml — every key at its schema default. Edit and pass via\n"
        "#   hif profile <model> <prompt> --config-file run.toml\n"
        "# Then confirm what will run:\n"
        "#   hif config show --config-file run.toml --diff\n"
        "# Reference: docs/CONFIG.md. Model name/backend always come from the\n"
        "# CLI arguments; the [model] name/backend lines here are ignored.\n\n"
        + body
    )
    console.print(f"[green]Wrote:[/green] {output}")


