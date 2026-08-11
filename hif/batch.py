"""`hif batch` — profile a workload file against one loaded model.

Loads the model/embedder (and optional teacher-forcing surrogate) exactly
once via SessionEngine, then streams one JSON dict per workload row to the
caller-supplied `emit` callback. Row failures never abort the stream: a
minimal error record is emitted and the run continues.

Error records carry the SAME `schema_version` as successful ones — the
constant is imported from hif.profile.record rather than restated here, so
one stream can never mix two schema versions.

Workload file: JSONL, one row per prompt:

    {"query_id": str, "text": str, "regime"?: str, "variants"?: [str, ...]}

`variants` carries researcher-authored perturbation paraphrases for the row;
when present they replace the configured generator pipeline for that row
(same rule as [perturbation] variants_file, which reads this same format).

The whole file is validated up front — a malformed line is a caller error
(WorkloadError), surfaced before any model is loaded.

Same as the engine: nothing is written implicitly, and per-row trace artifacts
are written only when the run opted in (--trace). A workload is the one place
where that default earns its keep on volume alone — one artifact per row.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hif.engine import SessionEngine
from hif.profile.record import RECORD_SCHEMA_VERSION



def _release_local_gpu_memory() -> None:
    """Best-effort MPS cache release between rows.

    A single row's `engine.profile_one` can run many generate() calls
    (baseline + perturbation variants + trajectory branches, each with its
    own KV cache) with no cleanup in between. Over a multi-row batch on a
    local MPS-backed model this accumulates — degrading row latency and,
    once the unified-memory ceiling is hit, stalling for hours rather than
    raising a clean OOM (observed: 4 clean rows, then a sharp per-row
    slowdown, then an indefinite hang on row 6 of a 10-row session). No-op
    for API backends and machines without MPS; a plain `import torch`
    failure is also treated as a no-op (hif runs without torch when only
    API backends are configured)."""
    try:
        import gc

        import torch
    except ImportError:
        return
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class WorkloadError(ValueError):
    """The workload file is unusable (missing, unreadable, malformed rows).

    Raised during up-front validation — before any model load — so the CLI
    can exit 3 without paying inference cost.
    """


@dataclass
class BatchRow:
    """One validated workload row."""

    query_id: str
    text: str
    regime: Optional[str] = None  # overrides the run-level default
    # Researcher-authored perturbation variants for this prompt. When present,
    # the row's profile uses these INSTEAD of the generator pipeline — the
    # same rule as [perturbation] variants_file, which reads this same row
    # format. None means "no opinion" (generators apply as configured); an
    # explicit [] is rejected at load, because "I authored zero variants" and
    # "use the generators" must not be spelled the same way.
    variants: Optional[list[str]] = None


def load_workload(path: Path, *, limit: Optional[int] = None) -> list[BatchRow]:
    """Parse and validate the whole workload file up front.

    Raises WorkloadError on a missing file or any malformed line — a
    half-valid workload silently profiling a subset is worse than no run.
    The whole file is validated before `limit` truncates it, so a malformed
    line past the limit still surfaces.
    """
    path = Path(path)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        raise WorkloadError(f"Workload file not found: {path}")
    except OSError as exc:
        raise WorkloadError(f"Could not read workload file {path}: {exc}")

    rows: list[BatchRow] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkloadError(f"{path}:{lineno}: not valid JSON: {exc}")
        if not isinstance(data, dict):
            raise WorkloadError(f"{path}:{lineno}: row must be a JSON object")
        query_id = data.get("query_id")
        text = data.get("text")
        if not isinstance(query_id, str) or not query_id:
            raise WorkloadError(
                f"{path}:{lineno}: missing/invalid \"query_id\" (non-empty string required)"
            )
        if not isinstance(text, str) or not text:
            raise WorkloadError(
                f"{path}:{lineno}: missing/invalid \"text\" (non-empty string required)"
            )
        regime = data.get("regime")
        if regime is not None and (not isinstance(regime, str) or not regime):
            raise WorkloadError(
                f"{path}:{lineno}: \"regime\" must be a non-empty string"
            )
        variants = data.get("variants")
        if variants is not None:
            if (
                not isinstance(variants, list)
                or not variants
                or not all(isinstance(v, str) and v.strip() for v in variants)
            ):
                raise WorkloadError(
                    f"{path}:{lineno}: \"variants\" must be a non-empty list of "
                    f"non-empty strings (omit the key to use the configured "
                    f"generators)"
                )
        rows.append(
            BatchRow(
                query_id=query_id,
                text=text,
                regime=regime,
                variants=variants,
            )
        )

    if limit is not None:
        rows = rows[: max(limit, 0)]
    return rows


def sanitize_query_id(query_id: str) -> str:
    """Filesystem-safe form of a query_id for trace artifact names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", query_id) or "row"


def _error_record(query_id: str, message: str) -> dict:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "query_id": query_id,
        "error": message,
    }


def _write_row_trace(engine, profile, row: BatchRow, *, seed: int,
                     trace_dir: Path) -> Path:
    """Persist the row's full profile artifact.

    Named with the (sanitized) query_id AND the content hash: two workload
    rows can share (model, prompt, seed) — the hash alone would collide and
    silently overwrite one row's artifact with another's.
    """
    from hif.profile.record import profile_hash
    from hif.profile.render_json import render_json

    h = profile_hash(engine.config.model.name, row.text, seed)
    path = Path(trace_dir) / f"profile_{sanitize_query_id(row.query_id)}_{h}.json"
    render_json(profile, path)
    return path


def run_batch(
    config,
    rows: list[BatchRow],
    *,
    default_regime: str = "batch",
    seed: int = 42,
    surrogate: bool = False,
    surrogate_model_id: Optional[str] = None,
    trace: bool = False,
    trace_dir: Optional[Path] = None,
    emit: Callable[[dict], None],
    include_units: bool = False,
    variant_io: bool = False,
    log: Callable[[str], None] = lambda _msg: None,
) -> tuple[int, int]:
    """Profile every row against one engine; stream records via `emit`.

    Returns (n_ok, n_failed). Row errors emit a minimal error record and
    continue — the stream never aborts mid-workload. `log` receives
    human-readable progress lines (the CLI routes them to stderr).
    """
    create_kwargs = dict(surrogate=surrogate)
    if surrogate_model_id is not None:
        create_kwargs["surrogate_model_id"] = surrogate_model_id
    engine = SessionEngine.create(config, **create_kwargs)

    resolved_trace_dir = Path(trace_dir) if trace_dir is not None else Path("traces")

    n_ok = 0
    n_failed = 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        regime = row.regime or default_regime
        t0 = time.perf_counter()
        try:
            # Per-row sink: variant continuations are held only for this
            # row's record, then dropped with the loop variable.
            variant_output_sink: Optional[dict] = {} if variant_io else None
            profile = engine.profile_one(
                row.text, regime=regime, seed=seed,
                authored_variants=row.variants,
                variant_output_sink=variant_output_sink,
            )
            elapsed = time.perf_counter() - t0

            trace_path: Optional[Path] = None
            if trace:
                trace_path = _write_row_trace(
                    engine, profile, row, seed=seed, trace_dir=resolved_trace_dir
                )

            extras: dict = {"query_id": row.query_id}
            if variant_io:
                from hif.perturbation.authored import variant_io_block

                extras["variant_io"] = variant_io_block(
                    profile, variant_output_sink or {}
                )
            record = engine.record_for(
                profile,
                prompt=row.text,
                regime=regime,
                seed=seed,
                latency={"pipeline": round(elapsed, 3)},
                trace_path=str(trace_path) if trace_path is not None else None,
                extras=extras,
                include_units=include_units,
            )
            emit(record)
            n_ok += 1
            log(f"[{i}/{total}] {row.query_id} ok ({elapsed:.1f}s)")
        except Exception as exc:  # noqa: BLE001 — row isolation is the contract
            elapsed = time.perf_counter() - t0
            message = str(exc) or exc.__class__.__name__
            emit(_error_record(row.query_id, message))
            n_failed += 1
            log(f"[{i}/{total}] {row.query_id} ERROR ({elapsed:.1f}s): {message}")
        finally:
            _release_local_gpu_memory()

    return n_ok, n_failed
