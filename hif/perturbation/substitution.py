"""Lexical-substitution perturbation generator: swaps domain-specific terms with alternatives."""

import logging
import random

from hif.perturbation.base import PerturbationGenerator, PerturbationResult

logger = logging.getLogger(__name__)

SUBSTITUTION_LISTS: dict[str, list[str]] = {
    # Nouns
    "person": ["individual", "human", "figure", "entity"],
    "people": ["individuals", "humans", "persons", "folks"],
    "thing": ["item", "object", "matter", "element"],
    "way": ["method", "approach", "manner", "means"],
    "time": ["moment", "period", "instance", "occasion"],
    "place": ["location", "area", "site", "spot"],
    "day": ["date", "period", "session", "instance"],
    "year": ["period", "span", "cycle", "term"],
    "problem": ["issue", "challenge", "difficulty", "matter"],
    "question": ["query", "inquiry", "matter", "issue"],
    "information": ["data", "details", "knowledge", "facts"],
    "system": ["framework", "structure", "arrangement", "setup"],
    "work": ["effort", "task", "labor", "activity"],
    "case": ["situation", "instance", "example", "scenario"],
    "point": ["aspect", "detail", "element", "factor"],
    # Verbs
    "is": ["remains", "appears", "seems", "stands"],
    "are": ["remain", "appear", "seem", "stand"],
    "make": ["create", "produce", "form", "generate"],
    "get": ["obtain", "acquire", "receive", "gain"],
    "go": ["proceed", "move", "advance", "continue"],
    "know": ["understand", "recognize", "realize", "grasp"],
    "think": ["believe", "consider", "suppose", "regard"],
    "use": ["employ", "apply", "utilize", "deploy"],
    "help": ["assist", "support", "aid", "facilitate"],
    "need": ["require", "demand", "necessitate", "want"],
    "feel": ["sense", "experience", "perceive", "notice"],
    "seem": ["appear", "look", "sound", "feel"],
    # Adjectives
    "good": ["excellent", "fine", "solid", "strong"],
    "bad": ["poor", "weak", "problematic", "difficult"],
    "important": ["significant", "critical", "essential", "key"],
    "different": ["distinct", "varied", "alternative", "diverse"],
    "large": ["substantial", "considerable", "significant", "major"],
    "small": ["minor", "limited", "modest", "slight"],
    "new": ["recent", "current", "modern", "fresh"],
    "old": ["prior", "previous", "earlier", "former"],
    "long": ["extended", "prolonged", "lengthy", "sustained"],
    "short": ["brief", "limited", "concise", "quick"],
    "right": ["correct", "appropriate", "suitable", "proper"],
    "wrong": ["incorrect", "inappropriate", "mistaken", "erroneous"],
    "high": ["elevated", "substantial", "significant", "considerable"],
    "low": ["minimal", "limited", "modest", "slight"],
    "early": ["initial", "preliminary", "prior", "preceding"],
    "late": ["subsequent", "later", "delayed", "recent"],
    # Adverbs
    "very": ["quite", "rather", "considerably", "substantially"],
    "often": ["frequently", "regularly", "commonly", "typically"],
    "always": ["consistently", "invariably", "constantly", "regularly"],
    "never": ["rarely", "seldom", "infrequently", "uncommonly"],
    "well": ["effectively", "properly", "successfully", "adequately"],
    "quickly": ["rapidly", "swiftly", "promptly", "efficiently"],
    "clearly": ["plainly", "obviously", "evidently", "explicitly"],
    "simply": ["merely", "purely", "solely", "straightforwardly"],
}


def _split_into_words(prompt: str) -> list[str]:
    """Simple whitespace split preserving token positions."""
    return prompt.split()


class SubstitutionGenerator(PerturbationGenerator):
    name = "substitution"

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        words = _split_into_words(prompt)
        rng = random.Random(seed)

        # Find all (index, word, alternatives) for words in the substitution dict
        candidates: list[tuple[int, str, list[str]]] = []
        for idx, word in enumerate(words):
            key = word.lower().rstrip(".,!?;:'\"")
            if key in SUBSTITUTION_LISTS:
                candidates.append((idx, word, SUBSTITUTION_LISTS[key]))

        if not candidates:
            logger.warning(
                "SubstitutionGenerator: no substitution target found in prompt %r; "
                "returning original text for all %d variants.",
                prompt,
                n_variants,
            )
            return PerturbationResult(
                original=prompt,
                variants=[prompt] * n_variants,
                generator=self.name,
            )

        variants: list[str] = []
        for _ in range(n_variants):
            new_words = list(words)
            idx, word, alts = rng.choice(candidates)
            replacement = rng.choice(alts)
            # Preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            # Preserve trailing punctuation
            suffix = ""
            stripped = word.rstrip(".,!?;:'\"")
            if len(stripped) < len(word):
                suffix = word[len(stripped):]
            new_words[idx] = replacement + suffix
            variants.append(" ".join(new_words))

        return PerturbationResult(original=prompt, variants=variants, generator=self.name)
