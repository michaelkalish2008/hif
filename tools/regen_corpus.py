"""Regenerate the ai-interpretability corpus on the current schema.

The published corpus is schema 0.1.0 — it predates the natural-units rewrite,
so it carries `metrics.stability.input_stability` (the removed `1 - x` score)
where the site's panels now promise `input_entropy_std_bits` in bits, and it
carries `findings.*_level` verdicts that no longer exist upstream.

One profile per (site model id, regime), using prompt[0] of each regime — the
same selection the 0.1.0 corpus used, so a regenerated file describes the same
run the old one did. Model and embedder load once per model, not once per run.

`raw_traces` is stripped before writing: it is the audit artifact (per-branch
and per-variant top-K distributions), nothing on the site reads it, and it is
~30% of the file.

Existing files are skipped, so an interrupted run resumes where it stopped.

    python3 tools/regen_corpus.py open  ../ai-interpretability/public/data
    python3 tools/regen_corpus.py api   ../ai-interpretability/public/data

The `api` plane needs credentials in the environment and bills per token.
Its input-side rows come from `--surrogate`: these backends cannot teacher-
force, so those quantities are read off the PROMPT by a small local model and
land in `prompt_measurements` with subject `prompt-only`. They are identical
across every model profiled on the same prompt and are not measurements of the
target — see docs/MEASUREMENTS.md § Subject.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

from hif.cli import _run_single_profile, _load_model, _load_embedder
from hif.cli_base import discover_env_files, load_env_file
from hif.config import RunConfig
from hif.profile.measure import measurements, prompt_measurements
from hif.prompts.suite import REGIMES

SEED = 42

# site model id -> (model name, backend, base_url or None, extra_body or None)
# Names and backends are what the deployed profiles record, so a regenerated
# file describes the same model the old one did.
OPEN = [
    ("reference_run",            "gpt2",                                     "hf", None, None),
    ("gpt2_medium",              "gpt2-medium",                              "hf", None, None),
    ("gemma_3_1b_it",            "google/gemma-3-1b-it",                     "hf", None, None),
    ("llama_3_2_1b",             "meta-llama/Llama-3.2-1B",                  "hf", None, None),
    ("qwen3_1_7b",               "Qwen/Qwen3-1.7B",                          "hf", None, None),
    ("deepseek_r1_distill_1_5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "hf", None, None),
]

# DeepSeek: `deepseek-chat` was retired — the account now serves only
# deepseek-v4-flash and deepseek-v4-pro, so the V3 readings in the published
# corpus cannot be reproduced and are not regenerated here. The v4 model gets
# its own id rather than inheriting `deepseek_v3`: publishing a V4 reading
# under a V3 label is the provenance error the subject rules exist to prevent.
#
# `thinking: disabled` is not a preference. Left enabled at max_new_tokens=64
# the model spends 48 tokens reasoning and returns 15 content steps, so its
# measured generation would be a quarter the length of every other model's.
DEEPSEEK_NO_THINKING = {"thinking": {"type": "disabled"}}

API = [
    ("gpt4_1_mini",       "gpt-4.1-mini",              "openai",    None, None),
    ("gpt4_1",            "gpt-4.1",                   "openai",    None, None),
    ("gpt5",              "gpt-5",                     "openai",    None, None),
    ("deepseek_v4_flash", "deepseek-v4-flash",         "openai",    "https://api.deepseek.com", DEEPSEEK_NO_THINKING),
    ("deepseek_v4_pro",   "deepseek-v4-pro",           "openai",    "https://api.deepseek.com", DEEPSEEK_NO_THINKING),
    ("gemini_2_5_flash",  "gemini-2.5-flash",          "gemini",    None, None),
    ("gemini_2_5_pro",    "gemini-2.5-pro",            "gemini",    None, None),
    ("claude_haiku_4_5",  "claude-haiku-4-5-20251001", "anthropic", None, None),
    ("claude_sonnet_4_6", "claude-sonnet-4-6",         "anthropic", None, None),
]


def run_plane(models, out: Path, *, surrogate: bool) -> int:
    embedder = _load_embedder()
    n_ok = n_fail = 0

    for site_id, name, backend, base_url, extra_body in models:
        print(f"\n=== {site_id}  ({name} / {backend}) ===", flush=True)

        base_config = None
        if base_url is not None or extra_body is not None:
            base_config = RunConfig()
            if base_url is not None:
                base_config.model.base_url = base_url
            if extra_body is not None:
                base_config.model.extra_body = extra_body

        try:
            # base_url/extra_body must be given HERE. The model is loaded once
            # per model and handed to every run, so anything set only on the
            # per-run config never reaches the client that makes the request.
            model = _load_model(name, backend, base_url=base_url, extra_body=extra_body)
        except Exception as exc:
            print(f"  LOAD FAILED: {exc}", flush=True)
            n_fail += len(REGIMES)
            continue

        dest = out / site_id
        dest.mkdir(parents=True, exist_ok=True)

        for regime in REGIMES:
            target = dest / f"{regime.name}.json"
            if target.exists():
                print(f"  {regime.name}: exists, skipping", flush=True)
                n_ok += 1
                continue
            t0 = time.time()
            try:
                profile, _ = _run_single_profile(
                    model_name=name,
                    prompt=regime.prompts[0],
                    regime=regime.name,
                    backend=backend,
                    seed=SEED,
                    output_dir=None,          # no markdown reports
                    max_new_tokens=64,
                    top_k=50,
                    model=model,
                    embedder=embedder,
                    n_perturbation_variants=5,  # audit-grade: 3 generators x 5
                    diagnostics=True,           # attention rows + semantic field
                    surrogate=surrogate,
                    base_config=base_config,
                )
            except Exception as exc:
                print(f"  {regime.name}: ERROR {exc}", flush=True)
                traceback.print_exc()
                n_fail += 1
                continue

            record = json.loads(profile.model_dump_json())
            record.pop("raw_traces", None)
            # The CLI's own answer, so a consumer never has to re-derive the
            # absence rules from `metrics.*`. `measurements` is about this
            # model; `prompt_measurements` is about the PROMPT under a
            # reference model and is identical across every model profiled on
            # it — see docs/MEASUREMENTS.md § Subject.
            record["measurements"] = measurements(profile)
            record["prompt_measurements"] = prompt_measurements(profile)
            # Write to a temp name first: a half-written file left by a crash
            # would otherwise be skipped as "exists" on the next run.
            tmp = target.with_suffix(".json.partial")
            tmp.write_text(json.dumps(record, separators=(",", ":")))
            tmp.rename(target)

            st = record.get("metrics", {}).get("stability", {}) or {}
            print(
                f"  {regime.name}: ok  {time.time() - t0:.0f}s  "
                f"{target.stat().st_size / 1e6:.1f}MB  "
                f"std={st.get('input_entropy_std_bits')}  "
                f"jsd={st.get('perturbation_jsd_bits')}",
                flush=True,
            )
            n_ok += 1

        del model

    print(f"\ndone: {n_ok} ok, {n_fail} failed -> {out}", flush=True)
    return 1 if n_ok == 0 else 0


def main() -> int:
    # Load credentials with hif's own dotenv parser rather than relying on the
    # caller to have sourced anything. `set -a; source .env` is not equivalent:
    # the shell parses the file as script, so a value containing `&`, `(` or a
    # space aborts it mid-way and silently leaves later variables unset. This
    # file has exactly such a line, and the symptom was a Vertex AI run failing
    # with "No credentials found" while `echo $GOOGLE_CLOUD_PROJECT` printed
    # fine in the same shell.
    for env_path in discover_env_files():
        n = load_env_file(env_path)
        print(f"loaded {n} variable(s) from {env_path}", flush=True)

    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    plane, out = sys.argv[1], Path(sys.argv[2])
    if plane == "open":
        return run_plane(OPEN, out, surrogate=False)
    if plane == "api":
        if not os.environ.get("OPENAI_API_KEY"):
            print("no credentials in environment — source the .env first")
            return 2
        return run_plane(API, out, surrogate=True)
    print(f"unknown plane {plane!r} — expected 'open' or 'api'")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
