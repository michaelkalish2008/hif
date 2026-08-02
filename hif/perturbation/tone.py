"""Tone perturbation generator: rewrites the seed prompt in varied registers."""

import re

from hif.perturbation.base import PerturbationGenerator, PerturbationResult

# Contraction expansions for formal register
_CONTRACTIONS: list[tuple[str, str]] = [
    (r"\bcan't\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bI'm\b", "I am"),
    (r"\bit's\b", "it is"),
    (r"\bdon't\b", "do not"),
    (r"\bdoesn't\b", "does not"),
    (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"),
    (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"),
    (r"\bhaven't\b", "have not"),
    (r"\bhasn't\b", "has not"),
    (r"\bhadn't\b", "had not"),
    (r"\bwould've\b", "would have"),
    (r"\bcould've\b", "could have"),
    (r"\bshould've\b", "should have"),
]

# Formal-word → casual replacements
_FORMAL_TO_CASUAL: list[tuple[str, str]] = [
    (r"\bhowever\b", "but"),
    (r"\btherefore\b", "so"),
    (r"\bconsequently\b", "so"),
    (r"\bfurthermore\b", "also"),
    (r"\bnevertheless\b", "still"),
    (r"\bregarding\b", "about"),
    (r"\bconcerning\b", "about"),
    (r"\bapproximately\b", "about"),
    (r"\bsufficient\b", "enough"),
    (r"\brequire\b", "need"),
    (r"\bpurchase\b", "buy"),
    (r"\bassist\b", "help"),
    (r"\butilize\b", "use"),
    (r"\binform\b", "tell"),
    (r"\binquire\b", "ask"),
]

# Hedging phrases to strip for "direct" register (anchored at start of sentence)
_HEDGES: list[str] = [
    "I think that",
    "I believe that",
    "It seems that",
    "It appears that",
    "Perhaps",
    "Maybe",
    "It is possible that",
    "One might say that",
]

_FORMAL_PREFIXES = ["Please note that", "It should be noted that"]
_HEDGED_PREFIXES = ["It is possible that", "One might consider that"]
_HEDGED_SUFFIXES = ["in some circumstances", "depending on the context"]


def _capitalize_pronoun_i(text: str) -> str:
    """Ensure standalone 'i' (the pronoun) is always uppercase."""
    return re.sub(r'\bi\b', 'I', text)


def _apply_contractions(text: str) -> str:
    for pattern, replacement in _CONTRACTIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _make_formal(prompt: str, idx: int) -> str:
    text = _apply_contractions(prompt)
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    prefix = _FORMAL_PREFIXES[idx % len(_FORMAL_PREFIXES)]
    # Avoid double-prefix
    if not text.lower().startswith(prefix.lower()):
        text = f"{prefix} {text[0].lower()}{text[1:]}"
    # Ensure standalone pronoun "i" is always uppercase
    text = _capitalize_pronoun_i(text)
    return text


def _make_casual(prompt: str) -> str:
    text = prompt
    for pattern, replacement in _FORMAL_TO_CASUAL:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _make_direct(prompt: str) -> str:
    text = prompt.strip()
    for hedge in _HEDGES:
        # Strip from beginning (case-insensitive)
        pattern = r"^" + re.escape(hedge) + r"\s*,?\s*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    # Capitalize after stripping
    if text:
        text = text[0].upper() + text[1:]
    return text


def _make_hedged(prompt: str, idx: int) -> str:
    prefix = _HEDGED_PREFIXES[idx % len(_HEDGED_PREFIXES)]
    suffix = _HEDGED_SUFFIXES[idx % len(_HEDGED_SUFFIXES)]
    text = prompt.strip()
    # Lower-case first char to flow after prefix
    if text:
        text = text[0].lower() + text[1:]
    # Strip trailing period to append suffix cleanly
    text = text.rstrip(".")
    return f"{prefix} {text} {suffix}."


# Four transformations cycled for n_variants
_TRANSFORMS = ["formal", "casual", "direct", "hedged"]


class ToneGenerator(PerturbationGenerator):
    name = "tone"

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        variants: list[str] = []
        for i in range(n_variants):
            transform = _TRANSFORMS[i % len(_TRANSFORMS)]
            if transform == "formal":
                variants.append(_make_formal(prompt, i))
            elif transform == "casual":
                variants.append(_make_casual(prompt))
            elif transform == "direct":
                variants.append(_make_direct(prompt))
            else:  # hedged
                variants.append(_make_hedged(prompt, i))
        return PerturbationResult(original=prompt, variants=variants, generator=self.name)
