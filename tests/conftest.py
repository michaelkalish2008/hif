"""Test-wide defaults.

`PerturbationConfig.paraphraser` defaults to "local", which loads a 4B
instruction-tuned checkpoint from the HuggingFace cache. That is the right
default for a real run and the wrong one for this suite: these tests exercise
pipeline plumbing — which measurements appear, which are absent, what the
record carries — not paraphrase quality. Inheriting the real default made the
suite ~6x slower (29s to 168s) and, worse, made it depend on weights a clean
CI machine has no reason to hold: the builder logs a warning and continues when
a generator fails, so a machine without them would produce zero variants and
fail on absent measurements rather than on anything the test is about.

Tests that care about the paraphraser set it themselves.
"""

import pytest

from hif.config import PerturbationConfig


# The shipped default, captured before this file overrides it. A test that
# wants to assert what a real run does must read this, not PerturbationConfig().
SHIPPED_PARAPHRASER = PerturbationConfig.model_fields["paraphraser"].default


def pytest_configure(config):  # noqa: ARG001
    """Pin the suite's paraphraser to the rule-based generators.

    Set once at collection rather than per-test: pydantic compiles a model's
    validator, so mutating the field default alone does not change what
    `PerturbationConfig()` produces — the rebuild is what makes it take.
    """
    PerturbationConfig.model_fields["paraphraser"].default = "rule"
    PerturbationConfig.model_rebuild(force=True)
    assert PerturbationConfig().paraphraser == "rule"
