"""Ambiguity perturbation generator: modifies lexical specificity of the seed prompt."""

import re

from hif.perturbation.base import PerturbationGenerator, PerturbationResult

# Each tuple is (specific, general/ambiguous).
# "increase" mode replaces specific → general (more ambiguous).
# "decrease" mode replaces general → specific (less ambiguous).
SPECIFICITY_LADDER: list[tuple[str, str]] = [
    ("physician", "doctor"),
    ("doctor", "person"),
    ("hospital", "facility"),
    ("facility", "place"),
    ("medication", "drug"),
    ("drug", "substance"),
    ("attorney", "lawyer"),
    ("lawyer", "professional"),
    ("automobile", "car"),
    ("car", "vehicle"),
    ("vehicle", "thing"),
    ("purchase", "buy"),
    ("buy", "get"),
    ("residence", "home"),
    ("home", "place"),
    ("employment", "job"),
    ("job", "work"),
    ("child", "person"),
    ("adult", "person"),
    ("immediately", "soon"),
    ("soon", "later"),
    ("always", "often"),
    ("often", "sometimes"),
    ("certain", "possible"),
    ("impossible", "difficult"),
]


def _replace_word(text: str, old: str, new: str) -> str:
    """Replace whole-word occurrences of *old* with *new* (case-insensitive match, preserve case)."""

    def _sub(m: re.Match) -> str:  # type: ignore[type-arg]
        matched = m.group(0)
        if matched.isupper():
            return new.upper()
        if matched[0].isupper():
            return new.capitalize()
        return new

    return re.sub(r"\b" + re.escape(old) + r"\b", _sub, text, flags=re.IGNORECASE)


class AmbiguityGenerator(PerturbationGenerator):
    name = "ambiguity"

    def __init__(self, mode: str = "increase") -> None:
        if mode not in ("increase", "decrease"):
            raise ValueError(f"mode must be 'increase' or 'decrease', got {mode!r}")
        self.mode = mode

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        if self.mode == "increase":
            # Replace specific words with more general/ambiguous ones
            applicable: list[tuple[str, str]] = [
                (specific, general)
                for specific, general in SPECIFICITY_LADDER
                if re.search(r"\b" + re.escape(specific) + r"\b", prompt, re.IGNORECASE)
            ]
        else:
            # Replace general words with more specific ones (reverse ladder)
            applicable = [
                (general, specific)
                for specific, general in SPECIFICITY_LADDER
                if re.search(r"\b" + re.escape(general) + r"\b", prompt, re.IGNORECASE)
            ]

        if not applicable:
            return PerturbationResult(
                original=prompt,
                variants=[prompt] * n_variants,
                generator=self.name,
            )

        variants: list[str] = []
        for i in range(n_variants):
            # Cycle through applicable replacements
            old, new = applicable[i % len(applicable)]
            variant = _replace_word(prompt, old, new)
            variants.append(variant)

        return PerturbationResult(original=prompt, variants=variants, generator=self.name)


# Legacy function shim
def generate_ambiguity_variants(prompt: str, n: int = 5) -> list[str]:
    """Return n prompt variants with injected lexical or structural ambiguity."""
    return AmbiguityGenerator().generate(prompt, n_variants=n).variants
