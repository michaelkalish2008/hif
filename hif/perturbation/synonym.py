"""Synonym-substitution perturbation generator: replaces content words with near-synonyms."""

import random
from typing import Optional

from hif.perturbation.base import PerturbationGenerator, PerturbationResult

# Auxiliary verbs that should never be substituted — their inflected forms carry
# grammatical meaning that WordNet synonyms do not preserve.
_AUX_VERBS = frozenset({
    "be", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    "shall", "must", "can",
})

def _is_common_word(word: str) -> bool:
    """Return True if *word* has a non-zero SemCor frequency in WordNet.

    WordNet stores corpus-derived usage counts on each lemma; archaic or
    highly obscure lemmas (e.g. "hebdomad", "eld") consistently have a count
    of 0.  Requiring count > 0 is a lightweight, dependency-free filter.
    """
    from nltk.corpus import wordnet as wn
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            if lemma.name().lower() == word.lower() and lemma.count() > 0:
                return True
    return False


def _ensure_nltk() -> None:
    """Lazy-download required NLTK resources on first use."""
    import nltk

    for resource in ("wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"):
        nltk.download(resource, quiet=True)


# Penn Treebank POS prefix → WordNet POS constant
_POS_MAP = {
    "NN": "n",  # noun
    "VB": "v",  # verb
    "JJ": "a",  # adjective
}


def _wn_pos(penn_tag: str) -> Optional[str]:
    """Map a Penn Treebank tag to a WordNet POS char, or None if not a target category."""
    for prefix, wn in _POS_MAP.items():
        if penn_tag.startswith(prefix):
            return wn
    return None


def _get_synonyms(word: str, wn_pos: str) -> list[str]:
    """Return a list of single-word synonyms different from *word*."""
    from nltk.corpus import wordnet as wn

    synonyms: list[str] = []
    for synset in wn.synsets(word, pos=wn_pos):
        for lemma in synset.lemmas():
            name = lemma.name()
            # Only single-word synonyms, different from the original, and common enough
            if "_" not in name and name.lower() != word.lower() and _is_common_word(name):
                synonyms.append(name)
    return list(dict.fromkeys(synonyms))  # deduplicate, preserving order


class SynonymGenerator(PerturbationGenerator):
    name = "synonym"

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        _ensure_nltk()
        import nltk

        tokens = nltk.word_tokenize(prompt)
        tagged = nltk.pos_tag(tokens)

        # Build list of (index, word, synonyms) for substitutable positions
        candidates: list[tuple[int, str, list[str]]] = []
        for idx, (word, tag) in enumerate(tagged):
            wn_pos = _wn_pos(tag)
            if wn_pos is None:
                continue
            # Skip auxiliary verbs — their inflected forms are grammatically fixed
            if word.lower() in _AUX_VERBS:
                continue
            syns = _get_synonyms(word, wn_pos)
            if syns:
                candidates.append((idx, word, syns))

        variants: list[str] = []
        rng = random.Random(seed)

        if not candidates:
            # Degenerate: no substitutions possible
            return PerturbationResult(
                original=prompt,
                # Degenerate: nothing to perturb. Produce NOTHING rather than
                # n copies of the prompt — each copy contributes a divergence
                # of exactly zero, so a prompt this generator cannot touch
                # would otherwise be recorded as a model that did not move.
                variants=[],
                generator=self.name,
            )

        for _ in range(n_variants):
            new_tokens = list(tokens)
            # Pick 1-2 positions to substitute
            n_subs = min(2, len(candidates))
            chosen = rng.sample(candidates, n_subs)
            for idx, word, syns in chosen:
                replacement = rng.choice(syns)
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_tokens[idx] = replacement
            variants.append(_rejoin(new_tokens))

        return PerturbationResult(original=prompt, variants=variants, generator=self.name)


def _rejoin(tokens: list[str]) -> str:
    """Rejoin tokens, re-attaching punctuation without leading space."""
    import string

    result: list[str] = []
    for tok in tokens:
        if tok in string.punctuation and result:
            result[-1] = result[-1] + tok
        else:
            result.append(tok)
    return " ".join(result)
