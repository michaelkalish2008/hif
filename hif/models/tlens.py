"""TransformerLens backend for mechanistic-friendly but black-box-compatible inference."""

from typing import Optional

try:
    import transformer_lens

    _TLENS_AVAILABLE = True
except ImportError:
    _TLENS_AVAILABLE = False

from hif.config import ModelConfig
from hif.models import chat_template as _chat_template
from hif.models.base import GenerationResult, Logits, Model, StepRecord, TopKEntry
from hif.utils.logging import get_logger

logger = get_logger(__name__)


class TLensModel(Model):
    """TransformerLens backend; exposes the same interface as HFModel but uses HookedTransformer.

    Requires the [tlens] extra:
        pip install hi[tlens]
    """

    def __init__(self, config: ModelConfig) -> None:
        if not _TLENS_AVAILABLE:
            raise ImportError(
                "TransformerLens is not installed. Install with: pip install hi[tlens]"
            )

        import torch

        self._config = config
        self._torch = torch

        logger.info(f"Loading TLens model: {config.name}")
        self._model = transformer_lens.HookedTransformer.from_pretrained(config.name)
        self._model.eval()
        logger.info(f"TLens model ready: {config.name}")

    # --- Properties ---

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def vocab_size(self) -> int:
        return self._model.cfg.d_vocab

    @property
    def context_length(self) -> int:
        return self._model.cfg.n_ctx

    @property
    def max_top_k(self) -> Optional[int]:
        return None  # unlimited

    @property
    def supports_teacher_forcing(self) -> bool:
        return True

    # --- What this checkpoint declares about chat framing ---

    @property
    def chat_template_present(self) -> Optional[bool]:
        tok = getattr(self._model, "tokenizer", None)
        if tok is None:
            return None
        return _chat_template.declares_chat_template(tok)

    @property
    def stops_on_chat_turn_end(self) -> Optional[bool]:
        """Tokenizer EOS only — HookedTransformer carries no generation config.

        So this misses the families whose chat-turn terminator lives only
        there (Gemma's `<end_of_turn>`). A miss is a notice that does not
        print, which is the direction hif/models/chat_template.py argues these
        should fail in.
        """
        tok = getattr(self._model, "tokenizer", None)
        if tok is None:
            return None
        return _chat_template.stops_on_chat_turn_end(tok, [tok.eos_token_id])

    # --- Tokenization ---

    def tokenize(self, text: str) -> list[int]:
        return self._model.to_tokens(text, prepend_bos=False)[0].tolist()

    def detokenize(self, ids: list[int]) -> str:
        import torch

        return self._model.to_string(torch.tensor(ids))

    # --- Forward pass ---

    def forward(self, input_ids: list[int]) -> Logits:
        import torch

        input_tensor = torch.tensor([input_ids])
        with torch.no_grad():
            logits = self._model(input_tensor)  # shape (1, seq_len, vocab_size)
        logits = logits[0]  # (seq_len, vocab_size)
        seq_len, vocab_size = logits.shape
        return Logits(
            values=logits.float().cpu().tolist(),
            seq_len=seq_len,
            vocab_size=vocab_size,
        )

    # --- Generation ---

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        top_k: int,
        seed: int,
    ) -> GenerationResult:
        """Step-by-step generation mirroring HFModel.generate() exactly."""
        import torch
        import torch.nn.functional as F

        # TLens does not expose a standard EOS; use None to never break early.
        running_ids: list[int] = list(input_ids)
        steps: list[StepRecord] = []

        for step in range(max_new_tokens):
            input_tensor = torch.tensor([running_ids])
            with torch.no_grad():
                logits_out = self._model(input_tensor)  # (1, seq_len, vocab_size)

            last_logits = logits_out[0, -1, :].float()

            logprobs = F.log_softmax(last_logits, dim=-1)
            probs = torch.exp(logprobs)

            k = min(top_k, last_logits.shape[0])
            topk_logits, topk_ids = torch.topk(last_logits, k)

            topk_entries = [
                TopKEntry(
                    token_id=int(topk_ids[i].item()),
                    token_str=self._model.to_string(
                        torch.tensor([int(topk_ids[i].item())])
                    ),
                    logit=float(topk_logits[i].item()),
                    logprob=float(logprobs[topk_ids[i]].item()),
                    prob=float(probs[topk_ids[i]].item()),
                )
                for i in range(k)
            ]

            # Deterministic per-step sampling
            step_seed = seed + step
            generator = torch.Generator()
            generator.manual_seed(step_seed)
            selected_idx = torch.multinomial(probs, num_samples=1, generator=generator)
            selected_token_id = int(selected_idx.item())
            selected_token_str = self._model.to_string(
                torch.tensor([selected_token_id])
            )

            steps.append(
                StepRecord(
                    step=step,
                    selected_token_id=selected_token_id,
                    selected_token_str=selected_token_str,
                    topk=topk_entries,
                )
            )

            running_ids.append(selected_token_id)

        generated_ids = running_ids[len(input_ids):]

        return GenerationResult(
            input_ids=list(input_ids),
            generated_ids=generated_ids,
            steps=steps,
            model_name=self.name,
            top_k=top_k,
            seed=seed,
        )
