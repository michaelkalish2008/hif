"""Researcher-authored perturbation variants, carried in the workload format.

The contract under test: variants travel in workload JSONL rows (`variants`
key) — the same format `hif batch` profiles. When present they replace the
generator pipeline; a prompt with no usable rows is a hard error; elicited
per-variant continuations surface in the record's `variant_io` block, never
back into the input file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hif.batch import WorkloadError, load_workload
from hif.config import (
    ClusterConfig,
    GenerationConfig,
    ModelConfig,
    PerturbationConfig,
    RunConfig,
    TrajectoryConfig,
)
from hif.perturbation.authored import (
    AuthoredVariantsError,
    load_authored_variants,
    variant_io_block,
)
from hif.profile.builder import build_profile

from profile_helpers import FakeEmbeddingModel, FakeModel

PROMPT = "hello world"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture()
def variants_jsonl(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "variants.jsonl",
        [
            {"query_id": "q1", "text": PROMPT, "variants": ["hi world", "hello, world"]},
            {"query_id": "q2", "text": "another prompt", "variants": ["unrelated"]},
        ],
    )


class TestWorkloadRowSchema:
    def test_variants_ride_on_batch_rows(self, variants_jsonl):
        rows = load_workload(variants_jsonl)
        assert rows[0].variants == ["hi world", "hello, world"]

    def test_omitted_variants_is_none_not_empty(self, tmp_path):
        """None = "use the generators"; the two must not be conflated."""
        path = _write_jsonl(tmp_path / "w.jsonl", [{"query_id": "q", "text": PROMPT}])
        assert load_workload(path)[0].variants is None

    def test_explicit_empty_variants_rejected_at_load(self, tmp_path):
        path = _write_jsonl(
            tmp_path / "w.jsonl", [{"query_id": "q", "text": PROMPT, "variants": []}]
        )
        with pytest.raises(WorkloadError, match="variants"):
            load_workload(path)

    def test_non_string_variant_rejected_at_load(self, tmp_path):
        path = _write_jsonl(
            tmp_path / "w.jsonl", [{"query_id": "q", "text": PROMPT, "variants": [1]}]
        )
        with pytest.raises(WorkloadError, match="variants"):
            load_workload(path)


class TestLoad:
    def test_matching_rows_in_file_order(self, variants_jsonl):
        assert load_authored_variants(variants_jsonl, PROMPT) == ["hi world", "hello, world"]

    def test_multiple_matching_rows_concatenate(self, tmp_path):
        path = _write_jsonl(
            tmp_path / "v.jsonl",
            [
                {"query_id": "a", "text": PROMPT, "variants": ["one"]},
                {"query_id": "b", "text": PROMPT, "variants": ["two"]},
            ],
        )
        assert load_authored_variants(path, PROMPT) == ["one", "two"]

    def test_no_matching_rows_is_a_hard_error(self, variants_jsonl):
        with pytest.raises(AuthoredVariantsError):
            load_authored_variants(variants_jsonl, "a prompt not in the file")

    def test_matched_rows_without_variants_still_error(self, tmp_path):
        """A matched row with no `variants` key is not a perturbation set."""
        path = _write_jsonl(tmp_path / "v.jsonl", [{"query_id": "q", "text": PROMPT}])
        with pytest.raises(AuthoredVariantsError):
            load_authored_variants(path, PROMPT)

    def test_missing_file_is_an_authored_error(self, tmp_path):
        with pytest.raises(AuthoredVariantsError, match="not found"):
            load_authored_variants(tmp_path / "absent.jsonl", PROMPT)

    def test_malformed_line_is_an_authored_error(self, tmp_path):
        path = tmp_path / "v.jsonl"
        path.write_text("not json\n")
        with pytest.raises(AuthoredVariantsError):
            load_authored_variants(path, PROMPT)


class TestBuilderIntegration:
    def _config(self) -> RunConfig:
        return RunConfig(
            model=ModelConfig(name="mock-model", backend="hf"),
            generation=GenerationConfig(max_new_tokens=3, top_k=5),
            trajectory=TrajectoryConfig(n_branches=0, rollout_steps=3),
            # Generators deliberately non-empty: authored variants must win
            # over them, not merge with them.
            perturbation=PerturbationConfig(n_variants=5, generators=["synonym", "tone"]),
            cluster=ClusterConfig(method="kmeans"),
        )

    def _build(self, config, variants, sink=None):
        return build_profile(
            model=FakeModel(vocab_size=100, context_length=512, name="mock-model"),
            prompt=PROMPT,
            regime="test",
            config=config,
            embedder=FakeEmbeddingModel(dim=16, seed=0),
            seed=42,
            authored_variants=variants,
            variant_output_sink=sink,
        )

    def test_authored_variants_replace_generators_entirely(self):
        profile = self._build(self._config(), ["hi world", "hello, world"])
        assert len(profile.perturbations) == 1
        record = profile.perturbations[0]
        assert record.generator == "authored"
        assert record.variants == ["hi world", "hello, world"]

    def test_sink_receives_one_continuation_per_variant(self):
        sink: dict[str, str] = {}
        self._build(self._config(), ["hi world", "hello, world"], sink=sink)
        assert set(sink) == {"hi world", "hello, world"}
        assert all(isinstance(v, str) for v in sink.values())

    def test_no_elicitation_means_empty_sink(self):
        """synthesized-input semantics: variants teacher-forced, never
        generated from — so there is nothing for variant_io to carry."""
        config = self._config()
        config.perturbation.elicit_variant_outputs = False
        sink: dict[str, str] = {}
        self._build(config, ["hi world"], sink=sink)
        assert sink == {}


class TestVariantIoBlock:
    def _profile_and_sink(self, elicit: bool = True):
        config = RunConfig(
            model=ModelConfig(name="mock-model", backend="hf"),
            generation=GenerationConfig(max_new_tokens=3, top_k=5),
            trajectory=TrajectoryConfig(n_branches=0, rollout_steps=3),
            perturbation=PerturbationConfig(
                n_variants=1, generators=[], elicit_variant_outputs=elicit
            ),
            cluster=ClusterConfig(method="kmeans"),
        )
        sink: dict[str, str] = {}
        profile = build_profile(
            model=FakeModel(vocab_size=100, context_length=512, name="mock-model"),
            prompt=PROMPT,
            regime="test",
            config=config,
            embedder=FakeEmbeddingModel(dim=16, seed=0),
            seed=42,
            authored_variants=["hi world", "hello, world"],
            variant_output_sink=sink,
        )
        return profile, sink

    def test_block_joins_inputs_to_elicited_outputs(self):
        profile, sink = self._profile_and_sink()
        block = variant_io_block(profile, sink)
        assert [e["input"] for e in block] == ["hi world", "hello, world"]
        assert all(e["generator"] == "authored" for e in block)
        assert all(isinstance(e["output"], str) for e in block)

    def test_unelicited_output_is_null_not_invented(self):
        profile, sink = self._profile_and_sink(elicit=False)
        block = variant_io_block(profile, sink)
        assert [e["output"] for e in block] == [None, None]
