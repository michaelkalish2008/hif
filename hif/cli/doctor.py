"""`hif doctor` — preflight: dependencies, services, credentials, readiness."""

from __future__ import annotations



from hif.cli._app import (
    app,
    console,
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
    # Charts: HTML needs plotly (a core dep); PNG additionally needs kaleido,
    # which is imported only inside fig.write_image() and so fails at WRITE
    # time, after a full pipeline has already run. Report it here instead.
    if core_ok:
        png_ok = _has("kaleido")
        console.print(
            f"  charts (--charts): [green]ok[/green] — HTML"
            + (", PNG" if png_ok else "")
            + ("" if png_ok else
               " [dim](PNG needs kaleido: pip install kaleido)[/dim]")
        )

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

    # Credentials. Each one names where it came from: the failure this command
    # exists to catch is a dotenv that was never read, and "unset" next to a
    # file the user believes they loaded is the whole diagnosis.
    from hif.cli._app import CREDENTIAL_VARS, USER_ENV_FILE, env_origin

    console.print("\n[bold]credentials[/bold]")
    for env in CREDENTIAL_VARS:
        origin = env_origin(env)
        if origin:
            console.print(f"  {env}: [green]set[/green] [dim]({origin})[/dim]")
        else:
            console.print(f"  {env}: [dim]unset[/dim]")
    if not any(os.environ.get(e) for e in CREDENTIAL_VARS):
        console.print(
            f"  [dim]None found. hif reads the nearest .env at or above the working "
            f"directory, then {USER_ENV_FILE}; --env-file names one explicitly. "
            f"Note that `source .env` on bare KEY=value lines sets shell variables, "
            f"not environment ones, so hif never sees them.[/dim]"
        )

    # Per-backend readiness
    console.print("\n[bold]backends[/bold]")
    dep_probe = {
        "hf": "transformers", "tlens": "transformer_lens",
        "ollama": "httpx", "openai": "openai", "anthropic": "anthropic",
        "gemini": "google.genai",
    }
    cred_probe = {
        "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
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
                  "`hif profile Qwen/Qwen3-0.6B-Base \"hello\" --backend hf`. "
                  "Run `hif models` for the full capability matrix.[/dim]\n")



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


