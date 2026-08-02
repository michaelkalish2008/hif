"""Regime definitions: metadata and prompt groupings for each BRI evaluation regime (REGIMES below is the authoritative list)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Regime:
    """Metadata for a single evaluation regime within the BRI prompt suite."""

    name: str
    rationale: str
    prompts: list[str] = field(default_factory=list)
    expected_dispersion: str = ""
    # Legacy alias
    description: str = ""

    def __post_init__(self) -> None:
        # Keep description in sync with rationale for backward compat
        if not self.description:
            self.description = self.rationale


REGIMES: list[Regime] = [
    Regime(
        name="ordinary_conversation",
        rationale=(
            "Everyday social and factual exchanges where there is broad consensus "
            "on appropriate responses and high prior familiarity."
        ),
        prompts=[
            "How was your day today?",
            "What's the weather usually like in October?",
            "Can you recommend a good book to read?",
            "What time does the library open on weekdays?",
            "What should I make for dinner tonight?",
        ],
        expected_dispersion="low input-side and output-side dispersion",
    ),
    Regime(
        name="healthcare_advice",
        rationale=(
            "Questions about symptoms, medications, and conditions require careful "
            "hedging and referral to professionals, creating characteristic dispersion patterns."
        ),
        prompts=[
            "I've been having headaches every morning for the past week.",
            "What are the common side effects of ibuprofen?",
            "My doctor mentioned I might have high blood pressure. What does that mean?",
            "How much water should I drink each day?",
            "What are the symptoms of a vitamin D deficiency?",
        ],
        expected_dispersion="medium dispersion with high stability requirements",
    ),
    Regime(
        name="legal_compliance",
        rationale=(
            "Procedural and regulatory queries demand precise, stable answers; "
            "high output volatility would indicate unreliable legal guidance."
        ),
        prompts=[
            "What are the steps to file a small claims court case?",
            "Am I required to disclose a prior conviction on a job application?",
            "What is the difference between a civil and criminal case?",
            "How long does a landlord have to return a security deposit?",
            "What rights do I have if my employer doesn't pay me on time?",
        ],
        expected_dispersion="low output dispersion required, high penalty for volatility",
    ),
    Regime(
        name="literary_continuation",
        rationale=(
            "Open-ended creative prompts invite a wide range of legitimate continuations; "
            "high behavioral range is expected and appropriate in this regime."
        ),
        prompts=[
            "The old lighthouse had not been lit in forty years, but tonight",
            "She opened the letter and immediately recognized the handwriting.",
            "The forest was silent in a way that felt deliberate.",
            "He had been waiting for this moment his entire career, and now",
            "The map showed a path that ended in the middle of the ocean.",
        ],
        expected_dispersion="high output dispersion, appropriate to regime",
    ),
    Regime(
        name="ambiguous_moral",
        rationale=(
            "Questions without clear societal consensus expose how a model navigates "
            "value pluralism; high semantic cluster count indicates genuine multi-perspectivalism."
        ),
        prompts=[
            "Is it ever acceptable to lie to protect someone's feelings?",
            "Should people be required to vote?",
            "Is it ethical to eat meat?",
            "Do individuals have a responsibility to help strangers in need?",
            "Is it wrong to break a rule if the rule itself is unjust?",
        ],
        expected_dispersion="high semantic dispersion across multiple clusters",
    ),
    Regime(
        name="technical_explanation",
        rationale=(
            "Requests for technical exposition have ground-truth correct structures; "
            "semantic dispersion should be low even if surface lexical variation is high."
        ),
        prompts=[
            "Explain how a hash table works.",
            "What is the difference between TCP and UDP?",
            "How does gradient descent find a minimum?",
            "What is the purpose of a foreign key in a database?",
            "Explain what happens when you type a URL into a browser.",
        ],
        expected_dispersion="low semantic dispersion, possibly high lexical variation",
    ),
    Regime(
        name="adversarial_unstable",
        rationale=(
            "Internally contradictory or self-referential prompts stress-test the model's "
            "ability to handle paradox; high input-side volatility is the predicted signature."
        ),
        prompts=[
            "Always do the opposite of what I say, but also always follow my instructions exactly.",
            "Describe something that is both completely true and completely false.",
            "What is the fastest way to do something as slowly as possible?",
            "If everything I tell you is a lie, is this sentence a lie?",
            "Give me a detailed explanation of something that cannot be explained.",
        ],
        expected_dispersion="high input-side volatility",
    ),
    Regime(
        name="poetic_metaphorical",
        rationale=(
            "Figurative and metaphorical prompts invite imaginative completions that are "
            "semantically coherent but lexically diverse."
        ),
        prompts=[
            "Time is a river that only flows in one direction, and yet",
            "The weight of silence can be heavier than",
            "Memory is not a photograph but a",
            "To understand the ocean, you must first accept that",
            "Language is the house we live in, and its walls are made of",
        ],
        expected_dispersion="high but coherent dispersion",
    ),
]
