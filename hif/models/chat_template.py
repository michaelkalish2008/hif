"""What a checkpoint declares about chat framing — and why hif does not act on it.

hif sends the prompt to the model as raw text. No chat template is applied on
any backend that exposes a tokenizer, and that is deliberate: the input-side
measurements teacher-force the model over the exact words the caller typed, so
wrapping them in `<|im_start|>user\\n…<|im_end|>` first would make every
input-side number a measurement of a string the caller never wrote. Applying a
template by default would change what every measurement means with nothing in
the record to say so.

The cost is paid on instruct-tuned checkpoints. The model reads the prompt as a
document to continue rather than a request to answer, and the continuation
reads as unrelated text — on `Qwen/Qwen3-0.6B` the sky-is-blue prompt comes
back as a run of unrelated questions. The measurements are still real
measurements of that continuation (`io_cosine_similarity` drops from 0.476 on
`Qwen/Qwen3-0.6B-Base` to 0.185 on `Qwen/Qwen3-0.6B`, which is the instrument
seeing exactly this), but a caller's first reading is that the tool is broken.
Hence the load-time notice in hif/cli/_load.py, which this module answers the
question behind.

Two different questions, and only one of them is a detector
-----------------------------------------------------------
**Does the checkpoint DECLARE a chat template?** `declares_chat_template` — a
literal fact about `tokenizer.chat_template`, no inference. This is what the
record carries (`provenance.chat_template_present`), because the record carries
facts.

**Does the checkpoint's own generation config STOP on a token its chat template
emits?** `stops_on_chat_turn_end` — an inference, and the one worth warning on.

They are not the same question, and the difference is why the notice is not
gated on the first one. `Qwen/Qwen3-0.6B-Base` ships a chat template in its own
`tokenizer_config.json` — so does `Qwen/Qwen2.5-0.5B` — and a notice gated on
"declares a template" fires on the base checkpoint in this project's own README
example, which is how a warning becomes something readers scroll past. Across
the fifteen checkpoints this was checked against, "declares a template"
mislabelled two base models as instruct-tuned; the EOS test mislabelled none.

The EOS test is an argument, not a pattern match on the name: a checkpoint
whose configured end-of-generation token is one its chat template emits between
turns was tuned to produce a chat turn and stop, which is the property that
makes raw continuation the wrong framing for it. It is family-agnostic — Qwen's
`<|im_end|>`, Gemma's `<end_of_turn>`, DeepSeek's `<｜end▁of▁sentence｜>` all
fall out of it with no per-family rule.

It fails toward silence, which is the direction that costs least. A repo that
ships no chat template at all cannot be caught by either test — the mirror
`NousResearch/Llama-2-7b-chat-hf` is one — and a missed notice leaves the
caller exactly where they were before this module existed. A false one spends
the notice's credibility on a correct run.
"""

from __future__ import annotations

from typing import Iterable, Optional

from hif.utils.logging import get_logger

logger = get_logger(__name__)

# One user turn and one assistant turn: enough for a template to emit whatever
# it puts between turns, which is the only part the EOS test reads. A prompt
# rendered with `add_generation_prompt=True` would stop before the assistant's
# terminator and miss it.
_PROBE_CONVERSATION = [
    {"role": "user", "content": "u"},
    {"role": "assistant", "content": "a"},
]


def declares_chat_template(tokenizer) -> bool:
    """Whether `tokenizer` carries a chat template. A literal fact, not a verdict.

    True on plenty of base checkpoints — see the module docstring. Read it as
    "the repo ships a template", never as "this checkpoint is instruct-tuned".
    """
    # transformers has carried this as a str and (for multi-template repos) as
    # a dict; both are truthy when populated and None when absent.
    return bool(getattr(tokenizer, "chat_template", None))


def stops_on_chat_turn_end(
    tokenizer, eos_token_ids: Iterable[Optional[int]]
) -> bool:
    """Whether the checkpoint stops generating on a token its template emits.

    `eos_token_ids` is every id the checkpoint would halt on — for a HF model
    that is `generation_config.eos_token_id` (an int or a list) unioned with
    the tokenizer's own. Gemma is why the generation config has to be in there:
    `google/gemma-3-1b-it` halts on `<end_of_turn>`, which appears only in the
    generation config; its tokenizer reports `<eos>`.

    False whenever the question cannot be answered — no template, a template
    that will not render, no EOS declared. An unanswerable question is not
    evidence for either answer, and the one caller is a notice that should stay
    quiet when it does not know.
    """
    if not declares_chat_template(tokenizer):
        return False

    ids = [i for i in eos_token_ids if i is not None]
    if not ids:
        return False

    try:
        rendered = tokenizer.apply_chat_template(
            _PROBE_CONVERSATION, tokenize=False
        )
    except Exception as exc:  # noqa: BLE001 — any template that will not render
        # A template can require tools, a system turn, or a role vocabulary
        # this probe does not supply. Debug, not warning: the caller is a
        # notice, and failing to produce a notice is not a fault the user can
        # act on.
        logger.debug("Chat template did not render for the EOS probe: %s", exc)
        return False

    if not isinstance(rendered, str):  # some templates return token lists
        return False

    for token_id in ids:
        try:
            token = tokenizer.convert_ids_to_tokens(token_id)
        except Exception:  # noqa: BLE001
            continue
        if token and token in rendered:
            return True
    return False
