"""Everything the CLI loads or probes before (and about) a run.

Backend resolution, the model / embedder / surrogate loads, and the live
catalogue probes `hif models` uses. Grouped because each one answers the same
question — what does this run actually get to work with — and because the
backend a name resolves to has to be answered identically everywhere it is
asked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from hif.cli._app import console, err_console


def _resolve_backend(model_name: str, backend: str, *, warn: bool = True) -> str:
    """The backend a (model name, --backend) pair actually runs on.

    Ollama-style names ("gemma3:4b-it-qat") contain a colon, which is invalid
    in HuggingFace repo ids — auto-route to the ollama backend rather than
    failing with an obscure repo-id validation error.

    `warn=False` is for callers that only need to know which backend the run
    will use (a capability guard, say) and would otherwise print the notice a
    second time before the load that actually performs the route.
    """
    if backend == "hf" and ":" in model_name:
        if warn:
            err_console.print(
                f"[yellow]{model_name!r} looks like an Ollama model tag — using "
                "--backend ollama. Pass --backend explicitly to override.[/yellow]"
            )
        return "ollama"
    return backend


def _load_model(
    model_name: str,
    backend: str,
    *,
    base_url: str | None = None,
    extra_body: dict | None = None,
):
    """Load a model. `base_url` and `extra_body` reach OpenAI-compatible
    providers that are not OpenAI.

    They are parameters rather than something the caller patches afterwards
    because the client is built here: without a base_url this constructs an
    OpenAI client and asks it for the named model, and a DeepSeek model name
    against OpenAI's endpoint answers `404 model_not_found` — an error that
    reads as "the provider retired the model" and is really "you asked the
    wrong provider". That cost three full regeneration passes before it was
    diagnosed.
    """
    # No dotenv load here. Credentials are resolved once, in the `hif`
    # callback, so that what `doctor` reports is what this load will get.
    from hif.config import ModelConfig
    backend = _resolve_backend(model_name, backend)
    from hif.models.factory import load_model
    try:
        model = load_model(ModelConfig(
            name=model_name, backend=backend,
            base_url=base_url, extra_body=extra_body,
        ))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    _notice_chat_tuned_checkpoint(model, model_name)
    return model


# Suffix rules that turn an instruct checkpoint's name into its base sibling's.
# Guesses, every one — which is why nothing below is printed until the Hub has
# confirmed the repo exists. Naming a repo that does not is worse than naming
# none: it sends the reader to a 404 in a message whose whole job is to be the
# way out.
#
# Ordered by how often they are right, and capped below at three lookups so a
# notice cannot turn into a series of round trips.
def _base_checkpoint_candidates(model_name: str) -> list[str]:
    """Plausible names for the base checkpoint `model_name` was tuned from."""
    org, _, name = model_name.rpartition("/")
    prefix = f"{org}/" if org else ""
    parts = name.split("-")
    candidates: list[str] = []

    # `Qwen2.5-7B-Instruct` → `Qwen2.5-7B`, `Mistral-7B-Instruct-v0.3` →
    # `Mistral-7B-v0.3`. A hyphen-delimited segment, so `Instructor` is safe.
    dropped = [p for p in parts if p.lower() not in ("instruct", "chat", "it")]
    if dropped and dropped != parts:
        candidates.append(prefix + "-".join(dropped))

    # Gemma names its base checkpoints `-pt` (pretrained) against the `-it`
    # instruction-tuned ones: `gemma-3-1b-it` → `gemma-3-1b-pt`.
    if parts[-1].lower() == "it":
        candidates.append(prefix + "-".join(parts[:-1] + ["pt"]))

    # Qwen3 puts the marker on the BASE checkpoint instead, so the instruct one
    # carries no suffix to strip: `Qwen3-0.6B` → `Qwen3-0.6B-Base`.
    candidates.append(f"{model_name}-Base")

    unique: list[str] = []
    for candidate in candidates:
        if candidate != model_name and candidate not in unique:
            unique.append(candidate)
    return unique[:3]


def _published_base_checkpoint(model_name: str) -> str | None:
    """The base sibling of `model_name` on the Hub, or None if none is found.

    Only reached for a checkpoint that already looks instruct-tuned, so a run
    on a base model makes no request at all — `hif profile
    Qwen/Qwen3-0.6B-Base …` stays the offline command the README says it is.

    Every failure answers None: offline, no `huggingface_hub`, an unreachable
    Hub, a slow one. The notice still prints; it just stops one sentence
    earlier. A network call is not allowed to decide whether the user hears
    about this.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.constants import HF_HUB_OFFLINE
    except ImportError:
        return None
    if HF_HUB_OFFLINE:
        return None

    api = HfApi()
    for candidate in _base_checkpoint_candidates(model_name):
        try:
            api.model_info(candidate, timeout=3.0)
        except Exception:  # noqa: BLE001 — missing, gated, offline, slow
            continue
        return candidate
    return None


def _notice_chat_tuned_checkpoint(model, model_name: str) -> None:
    """Say so when the checkpoint was tuned to answer and hif will not ask it to.

    hif continues the prompt as raw text on every backend, which on an
    instruct-tuned checkpoint produces a continuation rather than an answer —
    output a reader takes for a broken tool. The measurements are of that
    continuation and are real; what is wrong is the framing, and only the
    caller can fix it, so this is a notice on stderr rather than an error or a
    silent flag.

    Gated on the EOS test in hif/models/chat_template.py and NOT on "declares a
    chat template", which base checkpoints do too. That distinction is the
    whole reason this prints rarely enough to be worth reading.
    """
    if not getattr(model, "stops_on_chat_turn_end", None):
        return
    base = _published_base_checkpoint(model_name)
    err_console.print(
        f"[yellow]{model_name} looks instruct-tuned: it declares a chat "
        f"template and stops on a token that template emits. hif applies no "
        f"chat template — your prompt is continued as raw text, so the output "
        f"reads as a continuation rather than an answer, and the measurements "
        f"are of that continuation.[/yellow]"
        + (
            f"\n[yellow]Its base checkpoint is {base} — raw continuation is "
            f"the framing that one was trained for.[/yellow]"
            if base
            else ""
        )
    )


def _load_embedder():
    from hif.clustering.embed import EmbeddingModel
    from hif.config import EmbeddingConfig
    return EmbeddingModel(EmbeddingConfig())


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
    if name == "openai":
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
    # hf / tlens: any HF repo id is eligible, there's no fixed catalog.
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

