"""Word-order perturbation generator: produces grammatical reorderings of the seed prompt."""

import random
import string

from hif.perturbation.base import PerturbationGenerator, PerturbationResult

# Small hardcoded stop-word set (common English function words)
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "not",
        "no", "nor", "so", "yet", "both", "either", "neither", "each",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "their", "this", "that", "these", "those", "which",
        "who", "whom", "whose", "what", "where", "when", "why", "how",
        "if", "then", "than", "while", "after", "before", "since", "until",
        "about", "above", "below", "between", "into", "through", "during",
        "up", "down", "out", "over", "under", "again", "further", "once",
    }
)


def _is_content_word(tok: str) -> bool:
    """Return True if tok is a content word (not stop word, not punctuation)."""
    if tok in string.punctuation:
        return False
    return tok.lower() not in _STOP_WORDS


def _rejoin(tokens: list[str]) -> str:
    """Rejoin tokens, re-attaching punctuation without a leading space."""
    result: list[str] = []
    for tok in tokens:
        if tok in string.punctuation and result:
            result[-1] = result[-1] + tok
        else:
            result.append(tok)
    return " ".join(result)


def _ensure_nltk() -> None:
    import nltk

    for resource in ("punkt", "punkt_tab"):
        nltk.download(resource, quiet=True)


class WordOrderGenerator(PerturbationGenerator):
    name = "word_order"

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        _ensure_nltk()
        import nltk

        tokens = nltk.word_tokenize(prompt)

        # Positions of content words
        content_positions = [i for i, t in enumerate(tokens) if _is_content_word(t)]

        if len(content_positions) < 3:
            # Degenerate: not enough content words to swap meaningfully
            return PerturbationResult(
                original=prompt,
                variants=[prompt] * n_variants,
                generator=self.name,
            )

        # Identify pairs of adjacent content words (adjacent in content_positions list)
        adjacent_pairs: list[tuple[int, int]] = [
            (content_positions[k], content_positions[k + 1])
            for k in range(len(content_positions) - 1)
        ]

        variants: list[str] = []
        for i in range(n_variants):
            rng = random.Random(seed + i)
            new_tokens = list(tokens)
            pair = rng.choice(adjacent_pairs)
            pos_a, pos_b = pair
            new_tokens[pos_a], new_tokens[pos_b] = new_tokens[pos_b], new_tokens[pos_a]
            variants.append(_rejoin(new_tokens))

        return PerturbationResult(original=prompt, variants=variants, generator=self.name)


# Legacy function shim
def generate_word_order_variants(prompt: str, n: int = 5) -> list[str]:
    """Return n grammatical reorderings of the seed prompt."""
    return WordOrderGenerator().generate(prompt, n_variants=n).variants
