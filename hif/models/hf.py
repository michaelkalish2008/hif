"""HuggingFace Transformers backend for local model inference."""

from typing import Optional

import torch
import torch.nn.functional as F

from hif.config import ModelConfig
from hif.models import chat_template as _chat_template
from hif.models.base import GenerationResult, Logits, Model, StepRecord, TopKEntry
from hif.utils.logging import get_logger

logger = get_logger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFModel(Model):
    """HuggingFace backend wrapping AutoModelForCausalLM and AutoTokenizer."""

    def __init__(self, config: ModelConfig) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._config = config
        self._device = _resolve_device(config.device)
        self._dtype = _DTYPE_MAP.get(config.dtype, torch.float32)

        logger.info(f"Loading tokenizer: {config.name}")
        self._tokenizer = AutoTokenizer.from_pretrained(config.name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(
            f"Loading model: {config.name} | device={self._device} | dtype={config.dtype}"
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            config.name,
            torch_dtype=self._dtype,
        )
        self._model.eval()
        self._model.to(self._device)
        logger.info(f"Model ready: {config.name}")

    # --- Properties ---

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def context_length(self) -> int:
        return self._model.config.max_position_embeddings

    @property
    def max_top_k(self) -> Optional[int]:
        return None  # unlimited

    @property
    def supports_teacher_forcing(self) -> bool:
        return True

    # --- What this checkpoint declares about chat framing ---

    @property
    def chat_template_present(self) -> bool:
        return _chat_template.declares_chat_template(self._tokenizer)

    @property
    def stops_on_chat_turn_end(self) -> bool:
        return _chat_template.stops_on_chat_turn_end(
            self._tokenizer, self._eos_token_ids()
        )

    def _eos_token_ids(self) -> list[Optional[int]]:
        """Every id this checkpoint would halt generation on.

        The generation config first, because it is the one that carries the
        chat-turn terminator on some families (`google/gemma-3-1b-it` halts on
        `<end_of_turn>`; its tokenizer reports `<eos>`), and it holds either an
        int or a list.
        """
        configured = getattr(self._model.generation_config, "eos_token_id", None)
        ids = list(configured) if isinstance(configured, (list, tuple)) else [configured]
        ids.append(self._tokenizer.eos_token_id)
        return ids

    # --- Tokenization ---

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=True)

    def detokenize(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    # --- Forward pass ---

    def forward(self, input_ids: list[int]) -> Logits:
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            outputs = self._model(input_tensor)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
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
        eos_token_id = self._tokenizer.eos_token_id
        running_ids: list[int] = list(input_ids)
        steps: list[StepRecord] = []

        for step in range(max_new_tokens):
            input_tensor = torch.tensor(
                [running_ids], dtype=torch.long, device=self._device
            )
            with torch.no_grad():
                outputs = self._model(input_tensor)

            # Logits at the last position: shape (vocab_size,)
            last_logits = outputs.logits[0, -1, :].float()

            logprobs = F.log_softmax(last_logits, dim=-1)
            probs = torch.exp(logprobs)

            # Top-K entries by logit value
            k = min(top_k, last_logits.shape[0])
            topk_logits, topk_ids = torch.topk(last_logits, k)

            topk_entries = [
                TopKEntry(
                    token_id=int(topk_ids[i].item()),
                    token_str=self._tokenizer.decode([int(topk_ids[i].item())]),
                    logit=float(topk_logits[i].item()),
                    logprob=float(logprobs[topk_ids[i]].item()),
                    prob=float(probs[topk_ids[i]].item()),
                )
                for i in range(k)
            ]

            # Sample from full distribution with deterministic per-step seed.
            # Temperature applies to SAMPLING only — recorded logprobs/topk stay
            # raw, matching hosted-API semantics (OpenAI returns raw logprobs
            # regardless of temperature). None/1.0 = unchanged behavior.
            temp = self._config.temperature
            if temp is not None and temp != 1.0 and temp > 0:
                sample_probs = F.softmax(last_logits / temp, dim=-1)
            else:
                sample_probs = probs
            step_seed = seed + step
            generator = torch.Generator(device=self._device)
            generator.manual_seed(step_seed)
            selected_idx = torch.multinomial(sample_probs, num_samples=1, generator=generator)
            selected_token_id = int(selected_idx.item())
            selected_token_str = self._tokenizer.decode([selected_token_id])

            steps.append(
                StepRecord(
                    step=step,
                    selected_token_id=selected_token_id,
                    selected_token_str=selected_token_str,
                    topk=topk_entries,
                )
            )

            running_ids.append(selected_token_id)

            if eos_token_id is not None and selected_token_id == eos_token_id:
                break

        generated_ids = running_ids[len(input_ids):]

        return GenerationResult(
            input_ids=list(input_ids),
            generated_ids=generated_ids,
            steps=steps,
            model_name=self.name,
            top_k=top_k,
            seed=seed,
        )

    def get_attention(
        self,
        input_ids: list[int],
        max_new_tokens: int = 32,
        top_k: int = 50,
        seed: int = 42,
    ) -> dict:
        """Run generation then one forward pass with output_attentions=True on the full sequence."""
        from transformers import AutoModelForCausalLM

        # Step 1: generate without attention (fast)
        result = self.generate(input_ids, max_new_tokens=max_new_tokens, top_k=top_k, seed=seed)
        full_ids = list(input_ids) + result.generated_ids
        all_tokens = [self._tokenizer.decode([t]) for t in full_ids]

        # Step 2: single forward pass with eager attention (SDPA doesn't support output_attentions)
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self._device)
        eager_model = AutoModelForCausalLM.from_pretrained(
            self._config.name,
            torch_dtype=self._dtype,
            attn_implementation="eager",
        )
        eager_model.eval()
        eager_model.to(self._device)
        with torch.no_grad():
            outputs = eager_model(input_tensor, output_attentions=True)
        del eager_model

        attentions = outputs.attentions
        if not attentions:
            raise RuntimeError("Model did not return attention weights.")

        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]

        # Convert to [layer][head][query][key] nested lists
        attention_data = []
        for layer_attn in attentions:
            layer = layer_attn[0].float().cpu()
            heads = [layer[h].tolist() for h in range(n_heads)]
            attention_data.append(heads)

        return {
            "model": self.name,
            "tokens": all_tokens,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "attention": attention_data,
        }

    def get_attention_for_ids(self, full_ids: list[int], prompt_length: int) -> dict:
        """Run a single forward pass with output_attentions=True on a pre-specified token sequence.

        Use this instead of get_attention when the generated token sequence is already known
        (e.g. stored in a profile), so attention weights match the exact profile output.
        """
        from transformers import AutoModelForCausalLM

        all_tokens = [self._tokenizer.decode([t]) for t in full_ids]

        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self._device)
        eager_model = AutoModelForCausalLM.from_pretrained(
            self._config.name,
            torch_dtype=self._dtype,
            attn_implementation="eager",
        )
        eager_model.eval()
        eager_model.to(self._device)
        with torch.no_grad():
            outputs = eager_model(input_tensor, output_attentions=True)
        del eager_model

        attentions = outputs.attentions
        if not attentions:
            raise RuntimeError("Model did not return attention weights.")

        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]

        attention_data = []
        for layer_attn in attentions:
            layer = layer_attn[0].float().cpu()
            heads = [layer[h].tolist() for h in range(n_heads)]
            attention_data.append(heads)

        return {
            "model": self.name,
            "tokens": all_tokens,
            "prompt_length": prompt_length,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "attention": attention_data,
        }
