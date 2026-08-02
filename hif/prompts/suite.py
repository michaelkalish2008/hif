"""BRI prompt suite: curated seed prompts organized by evaluation regime."""

from __future__ import annotations

from hif.prompts.regimes import REGIMES, Regime

# Legacy dict-based interface
PROMPT_SUITE: dict[str, list[str]] = {r.name: r.prompts for r in REGIMES}

REGIME_NAMES: list[str] = [r.name for r in REGIMES]

_REGIME_MAP: dict[str, Regime] = {r.name: r for r in REGIMES}


def get_all_prompts() -> list[tuple[str, str]]:
    """Return list of (regime_name, prompt_text) for all regimes."""
    result: list[tuple[str, str]] = []
    for regime in REGIMES:
        for prompt in regime.prompts:
            result.append((regime.name, prompt))
    return result


def get_regime_prompts(regime_name: str) -> list[str]:
    """Return prompts for a specific regime."""
    return get_regime(regime_name).prompts


def get_regime(regime_name: str) -> Regime:
    """Return the Regime object for the given name.

    Raises ValueError if the regime is not found.
    """
    try:
        return _REGIME_MAP[regime_name]
    except KeyError:
        raise ValueError(
            f"Unknown regime: {regime_name!r}. "
            f"Available regimes: {REGIME_NAMES}"
        )
