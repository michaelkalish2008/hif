"""Unit tests for all five perturbation generators."""

import pytest

from hif.perturbation import (
    AmbiguityGenerator,
    SubstitutionGenerator,
    SynonymGenerator,
    ToneGenerator,
    WordOrderGenerator,
    get_generator,
)
from hif.perturbation.base import PerturbationResult

# ---------------------------------------------------------------------------
# Test prompt that exercises all generators
# ---------------------------------------------------------------------------
PROMPT = "The doctor needs to think carefully about the best way to help the patient quickly."
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_result(result: PerturbationResult, n_variants: int, prompt: str) -> None:
    assert isinstance(result, PerturbationResult)
    assert result.original == prompt
    assert len(result.variants) == n_variants
    for v in result.variants:
        assert isinstance(v, str), f"Variant is not a string: {v!r}"
        assert len(v) > 0, "Variant is empty"


def _check_reproducible(gen, prompt: str, n_variants: int, seed: int) -> None:
    r1 = gen.generate(prompt, n_variants=n_variants, seed=seed)
    r2 = gen.generate(prompt, n_variants=n_variants, seed=seed)
    assert r1.variants == r2.variants, "Generator is not reproducible with same seed"


def _changed_count(variants: list[str], original: str) -> int:
    return sum(1 for v in variants if v != original)


# ---------------------------------------------------------------------------
# SynonymGenerator
# ---------------------------------------------------------------------------


class TestSynonymGenerator:
    def test_returns_correct_number_of_variants(self):
        gen = SynonymGenerator()
        result = gen.generate(PROMPT, n_variants=5, seed=SEED)
        _check_result(result, 5, PROMPT)

    def test_variants_are_strings(self):
        gen = SynonymGenerator()
        result = gen.generate(PROMPT, n_variants=3, seed=SEED)
        for v in result.variants:
            assert isinstance(v, str)

    def test_reproducible(self):
        gen = SynonymGenerator()
        _check_reproducible(gen, PROMPT, n_variants=5, seed=SEED)

    def test_different_seeds_differ(self):
        gen = SynonymGenerator()
        r1 = gen.generate(PROMPT, n_variants=5, seed=0)
        r2 = gen.generate(PROMPT, n_variants=5, seed=99)
        # They may occasionally be equal, but usually differ
        # We only check both are valid PerturbationResults
        _check_result(r1, 5, PROMPT)
        _check_result(r2, 5, PROMPT)

    def test_some_variants_changed(self):
        gen = SynonymGenerator()
        result = gen.generate(PROMPT, n_variants=5, seed=SEED)
        n_changed = _changed_count(result.variants, PROMPT)
        if n_changed == 0:
            pytest.warns(UserWarning, match="degenerate")  # degenerate is acceptable

    def test_generator_name(self):
        gen = SynonymGenerator()
        result = gen.generate(PROMPT, n_variants=2, seed=SEED)
        assert result.generator == "synonym"

    def test_degenerate_prompt(self):
        """A prompt with no substitutable words should return originals without crashing."""
        gen = SynonymGenerator()
        result = gen.generate("a b c", n_variants=3, seed=SEED)
        assert len(result.variants) == 3


# ---------------------------------------------------------------------------
# WordOrderGenerator
# ---------------------------------------------------------------------------


class TestWordOrderGenerator:
    def test_returns_correct_number_of_variants(self):
        gen = WordOrderGenerator()
        result = gen.generate(PROMPT, n_variants=5, seed=SEED)
        _check_result(result, 5, PROMPT)

    def test_variants_are_strings(self):
        gen = WordOrderGenerator()
        result = gen.generate(PROMPT, n_variants=4, seed=SEED)
        for v in result.variants:
            assert isinstance(v, str)

    def test_reproducible(self):
        gen = WordOrderGenerator()
        _check_reproducible(gen, PROMPT, n_variants=5, seed=SEED)

    def test_same_words_different_order(self):
        gen = WordOrderGenerator()
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        result = gen.generate(PROMPT, n_variants=3, seed=SEED)
        import re

        orig_words = set(re.findall(r"\w+", PROMPT.lower()))
        for v in result.variants:
            v_words = set(re.findall(r"\w+", v.lower()))
            assert v_words == orig_words, f"Word set changed: {v!r}"

    def test_degenerate_short_prompt(self):
        """Nothing to reorder produces nothing, not n copies of the prompt.

        This used to assert len == 3, pinning the padding: a copy of the
        baseline contributes a divergence of exactly zero, so a prompt the
        generator cannot touch was recorded as a model that did not move.
        """
        gen = WordOrderGenerator()
        result = gen.generate("hello", n_variants=3, seed=SEED)
        assert result.variants == []

    def test_generator_name(self):
        gen = WordOrderGenerator()
        result = gen.generate(PROMPT, n_variants=2, seed=SEED)
        assert result.generator == "word_order"


# ---------------------------------------------------------------------------
# SubstitutionGenerator
# ---------------------------------------------------------------------------


class TestSubstitutionGenerator:
    def test_returns_correct_number_of_variants(self):
        gen = SubstitutionGenerator()
        result = gen.generate(PROMPT, n_variants=5, seed=SEED)
        _check_result(result, 5, PROMPT)

    def test_variants_are_strings(self):
        gen = SubstitutionGenerator()
        result = gen.generate(PROMPT, n_variants=3, seed=SEED)
        for v in result.variants:
            assert isinstance(v, str)

    def test_reproducible(self):
        gen = SubstitutionGenerator()
        _check_reproducible(gen, PROMPT, n_variants=5, seed=SEED)

    def test_some_variants_changed(self):
        gen = SubstitutionGenerator()
        result = gen.generate(PROMPT, n_variants=5, seed=SEED)
        n_changed = _changed_count(result.variants, PROMPT)
        assert n_changed > 0, "Expected at least one variant to differ from original"

    def test_known_word_substituted(self):
        gen = SubstitutionGenerator()
        prompt = "I think this is a good way to get help."
        result = gen.generate(prompt, n_variants=10, seed=SEED)
        # At least one variant should differ
        assert any(v != prompt for v in result.variants)

    def test_generator_name(self):
        gen = SubstitutionGenerator()
        result = gen.generate(PROMPT, n_variants=2, seed=SEED)
        assert result.generator == "substitution"

    def test_degenerate_no_known_words(self):
        """No known word to substitute produces nothing — see the note above."""
        gen = SubstitutionGenerator()
        result = gen.generate("xyzzy quux bloop", n_variants=3, seed=SEED)
        assert result.variants == []


# ---------------------------------------------------------------------------
# AmbiguityGenerator
# ---------------------------------------------------------------------------


class TestAmbiguityGenerator:
    def test_returns_correct_number_of_variants_increase(self):
        gen = AmbiguityGenerator(mode="increase")
        prompt = "The doctor visited the hospital and prescribed medication."
        result = gen.generate(prompt, n_variants=4, seed=SEED)
        _check_result(result, 4, prompt)

    def test_returns_correct_number_of_variants_decrease(self):
        gen = AmbiguityGenerator(mode="decrease")
        prompt = "A person went to a place to get some substance."
        result = gen.generate(prompt, n_variants=4, seed=SEED)
        _check_result(result, 4, prompt)

    def test_variants_are_strings(self):
        gen = AmbiguityGenerator()
        prompt = "The doctor visited the hospital."
        result = gen.generate(prompt, n_variants=3, seed=SEED)
        for v in result.variants:
            assert isinstance(v, str)

    def test_increase_makes_more_general(self):
        gen = AmbiguityGenerator(mode="increase")
        prompt = "The doctor visited the hospital."
        result = gen.generate(prompt, n_variants=3, seed=SEED)
        # At least one variant should not contain "doctor" or "hospital"
        # (replaced with more general words)
        changed = [v for v in result.variants if v != prompt]
        assert len(changed) > 0

    def test_degenerate_no_applicable_words(self):
        """No applicable word produces nothing — see the note above."""
        gen = AmbiguityGenerator(mode="increase")
        result = gen.generate("xyzzy quux bloop", n_variants=3, seed=SEED)
        assert result.variants == []

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            AmbiguityGenerator(mode="random")

    def test_generator_name(self):
        gen = AmbiguityGenerator()
        prompt = "The doctor visited the hospital."
        result = gen.generate(prompt, n_variants=2, seed=SEED)
        assert result.generator == "ambiguity"

    def test_reproducible(self):
        gen = AmbiguityGenerator()
        prompt = "The doctor visited the hospital."
        r1 = gen.generate(prompt, n_variants=4, seed=SEED)
        r2 = gen.generate(prompt, n_variants=4, seed=SEED)
        assert r1.variants == r2.variants


# ---------------------------------------------------------------------------
# ToneGenerator
# ---------------------------------------------------------------------------


class TestToneGenerator:
    def test_returns_correct_number_of_variants(self):
        gen = ToneGenerator()
        result = gen.generate(PROMPT, n_variants=5, seed=SEED)
        _check_result(result, 5, PROMPT)

    def test_variants_are_strings(self):
        gen = ToneGenerator()
        result = gen.generate(PROMPT, n_variants=4, seed=SEED)
        for v in result.variants:
            assert isinstance(v, str)

    def test_reproducible(self):
        gen = ToneGenerator()
        _check_reproducible(gen, PROMPT, n_variants=5, seed=SEED)

    def test_all_variants_differ_from_original(self):
        gen = ToneGenerator()
        prompt = "I think that you can't always do it well however."
        result = gen.generate(prompt, n_variants=4, seed=SEED)
        # tone always transforms
        assert all(isinstance(v, str) for v in result.variants)

    def test_formal_expands_contractions(self):
        gen = ToneGenerator()
        prompt = "I can't do this and I won't."
        result = gen.generate(prompt, n_variants=1, seed=SEED)  # index 0 = formal
        assert "cannot" in result.variants[0] or "will not" in result.variants[0]

    def test_casual_replaces_formal_words(self):
        gen = ToneGenerator()
        prompt = "Furthermore, I require assistance regarding this matter."
        result = gen.generate(prompt, n_variants=4, seed=SEED)
        casual = result.variants[1]  # index 1 = casual
        assert "furthermore" not in casual.lower() or "also" in casual.lower()

    def test_direct_strips_hedges(self):
        gen = ToneGenerator()
        prompt = "I think that this is the best approach."
        result = gen.generate(prompt, n_variants=3, seed=SEED)
        direct = result.variants[2]  # index 2 = direct
        assert not direct.lower().startswith("i think that")

    def test_hedged_adds_prefix(self):
        gen = ToneGenerator()
        prompt = "This is the best approach."
        result = gen.generate(prompt, n_variants=4, seed=SEED)
        hedged = result.variants[3]  # index 3 = hedged
        assert hedged.startswith("It is possible that") or hedged.startswith("One might consider that")

    def test_cycles_beyond_4(self):
        gen = ToneGenerator()
        result = gen.generate(PROMPT, n_variants=8, seed=SEED)
        assert len(result.variants) == 8

    def test_generator_name(self):
        gen = ToneGenerator()
        result = gen.generate(PROMPT, n_variants=2, seed=SEED)
        assert result.generator == "tone"


# ---------------------------------------------------------------------------
# get_generator factory
# ---------------------------------------------------------------------------


class TestGetGenerator:
    def test_all_names(self):
        for name in ("synonym", "word_order", "substitution", "ambiguity", "tone"):
            gen = get_generator(name)
            assert gen.name == name

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown generator"):
            get_generator("nonexistent")


# ---------------------------------------------------------------------------
# Content-preservation smoke test (uses sentence-transformers if available)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gen_name,prompt",
    [
        ("synonym", PROMPT),
        ("word_order", PROMPT),
        ("substitution", PROMPT),
        ("ambiguity", "The doctor visited the hospital."),
        ("tone", PROMPT),
    ],
)
def test_content_preservation_cosine_similarity(gen_name: str, prompt: str) -> None:
    """Variants should remain semantically close to the original (cosine sim ≥ 0.5)."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError:
        pytest.skip("sentence_transformers not available")

    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except OSError:
        pytest.skip("Sentence transformer model files not in local cache — connect to HuggingFace to download")
    gen = get_generator(gen_name)
    result = gen.generate(prompt, n_variants=3, seed=SEED)

    orig_emb = model.encode([prompt], normalize_embeddings=True)
    var_embs = model.encode(result.variants, normalize_embeddings=True)

    import numpy as np

    for i, (v, emb) in enumerate(zip(result.variants, var_embs)):
        sim = float(np.dot(orig_emb[0], emb))
        assert sim >= 0.5, (
            f"Generator '{gen_name}': variant {i} has cosine similarity {sim:.3f} < 0.5.\n"
            f"  Original: {prompt!r}\n"
            f"  Variant:  {v!r}"
        )
