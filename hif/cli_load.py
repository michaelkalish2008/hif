"""Everything the CLI loads or probes before (and about) a run.

Backend resolution, the model / embedder / surrogate loads, assembling a
multimodal input from image files, the validation corpus, and the live
catalogue probes `hif models` uses. Grouped because each one answers the same
question — what does this run actually get to work with — and because the
backend a name resolves to has to be answered identically everywhere it is
asked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from hif.cli_base import console, err_console


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
        return load_model(ModelConfig(
            name=model_name, backend=backend,
            base_url=base_url, extra_body=extra_body,
        ))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


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
