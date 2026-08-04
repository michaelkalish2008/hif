"""Unit tests for BRI prompt regimes and suite."""

from __future__ import annotations

import pytest


class TestRegimes:
    def test_all_regimes_present(self):
        from hif.prompts.suite import REGIME_NAMES

        expected = {
            "ordinary_conversation",
            "healthcare_advice",
            "legal_compliance",
            "literary_continuation",
            "ambiguous_moral",
            "technical_explanation",
            "adversarial_unstable",
            "poetic_metaphorical",
        }
        assert set(REGIME_NAMES) == expected

    def test_eight_regimes_total(self):
        from hif.prompts.suite import REGIME_NAMES

        assert len(REGIME_NAMES) == 8

    def test_each_regime_has_at_least_three_prompts(self):
        from hif.prompts.regimes import REGIMES

        for regime in REGIMES:
            assert len(regime.prompts) >= 3, (
                f"Regime {regime.name!r} has only {len(regime.prompts)} prompts"
            )

    def test_all_regimes_have_name(self):
        from hif.prompts.regimes import REGIMES

        for regime in REGIMES:
            assert regime.name, f"A regime has an empty name"

    def test_all_regimes_have_rationale(self):
        from hif.prompts.regimes import REGIMES

        for regime in REGIMES:
            assert regime.rationale, f"Regime {regime.name!r} has no rationale"

    def test_no_regime_declares_an_expected_result(self):
        """A regime carries a rationale, not an expectation.

        `expected_dispersion` used to sit here — a per-regime string like
        "low output dispersion required, high penalty for volatility". Nothing
        read it, but it was a threshold in waiting: surfaced next to a measured
        value it becomes a pass/fail, which is exactly what the `levels` block
        was removed for (see SIGNAL_SET_VERSION history in
        hif/profile/registry.py). The prompt suite is unlabeled by design; a
        field naming the right answer contradicts that.
        """
        from hif.prompts.regimes import REGIMES

        for regime in REGIMES:
            assert not hasattr(regime, "expected_dispersion"), (
                f"Regime {regime.name!r} declares an expected result"
            )


class TestSuiteFunctions:
    def test_get_regime_prompts_returns_nonempty_list(self):
        from hif.prompts.suite import get_regime_prompts

        prompts = get_regime_prompts("ordinary_conversation")
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_get_regime_prompts_all_strings(self):
        from hif.prompts.suite import get_regime_prompts

        for name in [
            "ordinary_conversation",
            "healthcare_advice",
            "literary_continuation",
            "adversarial_unstable",
        ]:
            prompts = get_regime_prompts(name)
            for p in prompts:
                assert isinstance(p, str)
                assert len(p) > 0

    def test_get_all_prompts_returns_tuples(self):
        from hif.prompts.suite import get_all_prompts

        result = get_all_prompts()
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            regime_name, prompt_text = item
            assert isinstance(regime_name, str)
            assert isinstance(prompt_text, str)

    def test_get_all_prompts_count(self):
        from hif.prompts.suite import get_all_prompts, REGIMES

        total_prompts = sum(len(r.prompts) for r in REGIMES)
        result = get_all_prompts()
        assert len(result) == total_prompts

    def test_get_all_prompts_regime_names_valid(self):
        from hif.prompts.suite import get_all_prompts, REGIME_NAMES

        for regime_name, _ in get_all_prompts():
            assert regime_name in REGIME_NAMES

    def test_get_regime_returns_regime_object(self):
        from hif.prompts.suite import get_regime
        from hif.prompts.regimes import Regime

        r = get_regime("poetic_metaphorical")
        assert isinstance(r, Regime)
        assert r.name == "poetic_metaphorical"

    def test_unknown_regime_raises_value_error(self):
        from hif.prompts.suite import get_regime

        with pytest.raises(ValueError, match="nonexistent"):
            get_regime("nonexistent")

    def test_unknown_regime_raises_for_empty_string(self):
        from hif.prompts.suite import get_regime

        with pytest.raises(ValueError):
            get_regime("")

    def test_regime_names_list_exported(self):
        from hif.prompts.suite import REGIME_NAMES

        assert isinstance(REGIME_NAMES, list)
        assert all(isinstance(n, str) for n in REGIME_NAMES)

    def test_prompt_suite_dict_exported(self):
        from hif.prompts.suite import PROMPT_SUITE

        assert isinstance(PROMPT_SUITE, dict)
        assert "ordinary_conversation" in PROMPT_SUITE
        assert isinstance(PROMPT_SUITE["ordinary_conversation"], list)

    def test_each_regime_prompts_are_non_empty_strings(self):
        from hif.prompts.suite import get_all_prompts

        for regime_name, prompt_text in get_all_prompts():
            assert prompt_text.strip(), (
                f"Empty prompt in regime {regime_name!r}"
            )
