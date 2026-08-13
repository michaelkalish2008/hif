"""Chat framing: what the checkpoint declares, and what hif says about it.

hif applies no chat template on any backend — the prompt is continued as raw
text. On an instruct-tuned checkpoint that produces a continuation rather than
an answer, which reads as a broken tool. Two things exist to keep that
legible: `provenance.chat_template_present` in the record, and a load-time
notice.

They are gated on DIFFERENT questions, and the tests below exist mostly to pin
that apart. `Qwen/Qwen3-0.6B-Base` ships a chat template in its own
tokenizer_config.json — so a notice gated on "declares a template" would fire
on the base checkpoint in this project's README example. The fixtures here are
built from the real declarations of the checkpoints named in their comments.
"""

from __future__ import annotations

import pytest

from hif.cli import _load
from hif.models.chat_template import declares_chat_template, stops_on_chat_turn_end


class FakeTokenizer:
    """A tokenizer's chat-template surface and nothing else."""

    def __init__(self, *, template=None, rendered="", eos_token_id=None, tokens=None):
        self.chat_template = template
        self.eos_token_id = eos_token_id
        self._rendered = rendered
        self._tokens = tokens or {}

    def apply_chat_template(self, conversation, tokenize=False):
        if self._rendered is None:
            raise ValueError("this template requires a system turn")
        return self._rendered

    def convert_ids_to_tokens(self, token_id):
        return self._tokens.get(token_id)


# The real declarations, as read off the cached checkpoints. Both Qwen rows
# carry the SAME template — the difference between them is which token the
# checkpoint stops on, which is the entire point.
QWEN_TEMPLATE = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
QWEN_RENDERED = "<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n"
QWEN_TOKENS = {151643: "<|endoftext|>", 151645: "<|im_end|>"}


def _qwen_base() -> FakeTokenizer:
    """Qwen/Qwen3-0.6B-Base — a template, stopping on <|endoftext|>."""
    return FakeTokenizer(
        template=QWEN_TEMPLATE, rendered=QWEN_RENDERED,
        eos_token_id=151643, tokens=QWEN_TOKENS,
    )


def _qwen_instruct() -> FakeTokenizer:
    """Qwen/Qwen3-0.6B — the same template, stopping on <|im_end|>."""
    return FakeTokenizer(
        template=QWEN_TEMPLATE, rendered=QWEN_RENDERED,
        eos_token_id=151645, tokens=QWEN_TOKENS,
    )


def _gemma_it() -> FakeTokenizer:
    """google/gemma-3-1b-it — the turn terminator is in the GENERATION config.

    Its tokenizer reports <eos>; only generation_config.eos_token_id carries
    <end_of_turn>, which is the token the template actually emits.
    """
    return FakeTokenizer(
        template="{{ '<start_of_turn>' }}",
        rendered="<start_of_turn>user\nu<end_of_turn>\n<start_of_turn>model\na<end_of_turn>\n",
        eos_token_id=1, tokens={1: "<eos>", 106: "<end_of_turn>"},
    )


# ---------------------------------------------------------------------------
# declares_chat_template — a literal fact, and not a detector
# ---------------------------------------------------------------------------


def test_declares_is_true_for_a_base_checkpoint_that_ships_a_template():
    # The whole reason the notice is not gated on this function.
    assert declares_chat_template(_qwen_base()) is True


def test_declares_is_false_without_one():
    assert declares_chat_template(FakeTokenizer()) is False


def test_declares_handles_a_dict_of_named_templates():
    # transformers has carried multi-template repos as a dict.
    assert declares_chat_template(FakeTokenizer(template={"default": "x"})) is True


# ---------------------------------------------------------------------------
# stops_on_chat_turn_end — the inference the notice IS gated on
# ---------------------------------------------------------------------------


def test_base_checkpoint_with_a_template_does_not_stop_on_a_turn_end():
    # <|endoftext|> is not a token the template emits. This is the case that
    # would make a template-presence notice fire on the README's own model.
    assert stops_on_chat_turn_end(_qwen_base(), [151643]) is False


def test_instruct_checkpoint_stops_on_a_turn_end():
    assert stops_on_chat_turn_end(_qwen_instruct(), [151645, 151643]) is True


def test_generation_config_eos_is_what_catches_gemma():
    tok = _gemma_it()
    # The tokenizer's own EOS alone misses it — which is what the tlens
    # backend is limited to, and why it under-warns.
    assert stops_on_chat_turn_end(tok, [1]) is False
    assert stops_on_chat_turn_end(tok, [1, 106]) is True


def test_no_template_is_false_whatever_the_eos():
    assert stops_on_chat_turn_end(FakeTokenizer(eos_token_id=50256), [50256]) is False


def test_no_declared_eos_is_false():
    assert stops_on_chat_turn_end(_qwen_instruct(), [None]) is False


def test_a_template_that_will_not_render_is_false():
    tok = FakeTokenizer(
        template="{{ tools[0] }}", rendered=None,
        eos_token_id=151645, tokens=QWEN_TOKENS,
    )
    # Unanswerable, so it answers no — a notice that cannot tell must stay quiet.
    assert stops_on_chat_turn_end(tok, [151645]) is False


# ---------------------------------------------------------------------------
# Naming the base sibling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name,expected",
    [
        # Qwen3 marks the BASE checkpoint, so there is no suffix to strip.
        ("Qwen/Qwen3-0.6B", ["Qwen/Qwen3-0.6B-Base"]),
        # Gemma's base checkpoints are -pt. The bare strip is tried first and
        # simply does not resolve.
        ("google/gemma-3-1b-it",
         ["google/gemma-3-1b", "google/gemma-3-1b-pt", "google/gemma-3-1b-it-Base"]),
        ("HuggingFaceTB/SmolLM2-135M-Instruct",
         ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct-Base"]),
        # The marker is a segment, not a substring: the version suffix stays.
        ("mistralai/Mistral-7B-Instruct-v0.3",
         ["mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3-Base"]),
        ("gpt2", ["gpt2-Base"]),
    ],
)
def test_base_checkpoint_candidates(model_name, expected):
    assert _load._base_checkpoint_candidates(model_name) == expected


def test_candidates_are_capped(monkeypatch):
    # A notice must not become a series of round trips.
    assert len(_load._base_checkpoint_candidates("org/model-Instruct-it")) <= 3


def test_no_lookup_when_the_hub_is_offline(monkeypatch):
    import huggingface_hub.constants as constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    def explode(*a, **k):  # pragma: no cover — the point is that it is not called
        raise AssertionError("the Hub was queried while offline")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", explode)
    assert _load._published_base_checkpoint("Qwen/Qwen3-0.6B") is None


# ---------------------------------------------------------------------------
# The notice
# ---------------------------------------------------------------------------


class FakeModel:
    def __init__(self, *, stops: bool | None):
        self.stops_on_chat_turn_end = stops


@pytest.fixture()
def no_hub(monkeypatch):
    """No network from the notice path — every test here decides the answer."""
    monkeypatch.setattr(_load, "_published_base_checkpoint", lambda name: None)


def test_notice_is_silent_on_a_base_checkpoint(no_hub, capsys):
    _load._notice_chat_tuned_checkpoint(FakeModel(stops=False), "Qwen/Qwen3-0.6B-Base")
    assert capsys.readouterr().err == ""


def test_notice_is_silent_when_the_backend_cannot_answer(no_hub, capsys):
    # Hosted APIs and Ollama: None is "not asked", and a notice is not a guess.
    _load._notice_chat_tuned_checkpoint(FakeModel(stops=None), "gpt-4o")
    assert capsys.readouterr().err == ""


def test_notice_names_the_model_and_says_what_hif_did(no_hub, capsys):
    _load._notice_chat_tuned_checkpoint(FakeModel(stops=True), "Qwen/Qwen3-0.6B")
    err = " ".join(capsys.readouterr().err.split())
    assert "Qwen/Qwen3-0.6B" in err
    assert "applies no chat template" in err
    # It must not read as "the measurements are wrong" — they are measurements
    # of the continuation, which is a different statement.
    assert "measurements are of that continuation" in err


def test_notice_names_a_confirmed_base_checkpoint(monkeypatch, capsys):
    monkeypatch.setattr(
        _load, "_published_base_checkpoint", lambda name: "Qwen/Qwen3-0.6B-Base"
    )
    _load._notice_chat_tuned_checkpoint(FakeModel(stops=True), "Qwen/Qwen3-0.6B")
    assert "Qwen/Qwen3-0.6B-Base" in " ".join(capsys.readouterr().err.split())


def test_notice_omits_the_sentence_when_no_base_is_found(no_hub, capsys):
    _load._notice_chat_tuned_checkpoint(FakeModel(stops=True), "org/only-child")
    err = " ".join(capsys.readouterr().err.split())
    assert "applies no chat template" in err
    assert "base checkpoint" not in err


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_provenance_records_the_declaration():
    from hif.profile.provenance import RunProvenance

    prov = RunProvenance(generation_model="m", chat_template_present=True)
    assert prov.model_dump()["chat_template_present"] is True


def test_provenance_defaults_to_not_asked():
    from hif.profile.provenance import RunProvenance

    # None, never False: a backend with no tokenizer of the checkpoint's own
    # has not answered "no template", it has not been asked.
    assert RunProvenance(generation_model="m").chat_template_present is None


@pytest.mark.parametrize("declared", [True, False, None])
def test_builder_records_whatever_the_backend_answered(declared):
    from hif.profile.builder import _run_provenance
    from tests.unit.mock_backends import alpha_model
    from tests.unit.profile_helpers import _make_profile

    model = alpha_model()
    if declared is not None:
        # A backend that CAN answer; the mock (like every hosted backend)
        # inherits None from Model and answers nothing.
        model.__class__.chat_template_present = property(lambda self: declared)
    try:
        reference = _make_profile()
        prov = _run_provenance(
            model=model,
            input_teacher_forcing_model=model.name,
            output_distribution_model=model.name,
            attention_analysis=None,
            output_trace=reference.output_side,
            trajectory=reference.trajectory,
        )
        assert prov.chat_template_present is declared
    finally:
        if declared is not None:
            del model.__class__.chat_template_present
