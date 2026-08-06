"""Hermeneutic Attention — bidirectional text analysis for hif.

Methodological position
-----------------------
Hermeneutic Attention uses a bidirectional encoder (DistilBERT by default) as a
reading instrument applied to texts as objects.  Three readings are performed:

1. **Reading the input**: the prompt is analyzed on its own terms — which tokens
   carry structural weight, how they attend to each other.  Perturbing tokens and
   watching the attention pattern shift reveals which tokens are load-bearing,
   without ever looking inside the generation model.

2. **Reading the output**: the generated continuation is analyzed independently —
   its own internal attention structure, without reference to how it was produced.
   No concatenation, no joint forward pass.

3. **Resonance comparison**: the two readings are compared analytically.  Which
   continuation tokens echo the load-bearing structure of the input?  High
   resonance = the output is hermeneutically anchored to the input's semantic
   structure.  Low resonance = free-floating.

4. **Joint trajectory trace**: DistilBERT is run on [prompt + continuation[:k]]
   at regular intervals as the continuation grows.  The cross-attention block
   (continuation tokens → prompt tokens) tracks how prompt tokens hold or release
   their grip as the continuation unfolds.  Fading tokens lose cross-attention
   share over time; persistent tokens remain load-bearing throughout; emerging
   pivots gain prominence as the continuation develops its own semantic momentum.

This is interpretation in the hermeneutic sense: we read each text, then hold
them together and ask where they resonate.  The generation process is not
observed; what is observed is the semantic relationship between input and output
as texts.

The analysis model (DistilBERT by default) plays a distinct third role —
separate from both the model under analysis (GPT-2) and the embedding model
(MiniLM/EmbeddingGemma).  GPT-2's generation mechanism is never exposed; its
attention weights are never accessed.

Implementation notes
--------------------
The Pydantic schemas defined here (``TextAttentionAnalysis``, ``AttentionMap``,
``InputAttentionAnalysis``, etc.) are the data contract consumed by all three
plot functions in ``hi.plots.attention``.  They are accessed by named attribute,
not as dicts — do not delete them.

They are not imported at the top level of ``hi.profile.schema`` because this
module loads transformers and torch lazily (inside ``AttentionAnalyzer.__init__``
via ``self._tokenizer`` / ``self._model``).  A top-level import would force those
heavy dependencies to load at schema-import time, which every module touching
``BehavioralRangeProfile`` would pay.  The ``TYPE_CHECKING`` guard in schema.py
keeps the type visible to static analysers without triggering the runtime import.
``BehavioralRangeProfile.attention`` is typed ``Optional[Any]``; its runtime type
is always ``TextAttentionAnalysis | None``.

The individual public methods (``analyze_input``, ``analyze_continuation``,
``compare``, ``analyze_joint_trajectory``) are not called directly from outside
this module — all external access enters through the single ``AttentionAnalyzer.analyze()``
entry point in ``hi.profile.builder``.  Their apparent greyness in editors is a
reachability false-positive, not dead code.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from hif.config import AttentionConfig
from hif.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TokenImportance(BaseModel):
    """Attention-based importance of a single token."""

    token_str: str
    token_idx: int
    importance: float  # normalized sum of attention received from all other tokens


class AttentionMap(BaseModel):
    """Aggregated attention weights over a token sequence.

    Produced by a bidirectional encoder analyzing text as an object.
    Not derived from the model under analysis — the generation mechanism
    is not accessed.
    """

    tokens: list[str]
    weights: list[list[float]]  # (n_tokens, n_tokens) aggregated attention matrix
    token_importance: list[TokenImportance]
    analysis_model: str
    aggregate_method: str


class AttentionDelta(BaseModel):
    """Change in attention structure when one input token is perturbed."""

    original_token_str: str
    perturbed_token_str: str
    token_idx: int
    importance_deltas: list[float]  # per-token change in importance score
    most_affected: list[tuple[str, float]]  # (token_str, delta) for top-5 most changed


class InputAttentionAnalysis(BaseModel):
    """Attention-based structural analysis of the input prompt.

    Shows which tokens are load-bearing and how perturbations shift
    the attention structure — without accessing the generation model's internals.
    """

    prompt_text: str
    attention_map: AttentionMap
    perturbation_deltas: list[AttentionDelta]  # one per perturbed variant analyzed


class TokenResonance(BaseModel):
    """Resonance of one output token with the input's load-bearing structure."""

    token_str: str
    token_idx: int
    resonance_score: float      # [0, 1] — 1 = strongly echoes an important input token
    anchored_to: str            # which input token it most resonates with ("" if none)


class HermeneuticComparison(BaseModel):
    """Interpretive comparison of input and output attention structures.

    Neither concatenation nor cross-attention from a joint forward pass.
    Two independent readings, compared analytically — hermeneutic resonance.
    """

    prompt_attention: AttentionMap
    continuation_attention: AttentionMap
    token_resonance: list[TokenResonance]   # one per continuation token
    mean_resonance: float                   # overall anchoring of output to input
    free_floating_tokens: list[str]         # continuation tokens with resonance < 0.2
    anchored_tokens: list[str]              # continuation tokens with resonance > 0.5


class AttentionCheckpoint(BaseModel):
    """Cross-attention state at a single point in the growing continuation.

    Produced by running DistilBERT on [prompt + continuation[:k]] and extracting
    the cross-block: how continuation tokens attend to prompt tokens at this step.
    """

    step: int                                   # number of continuation tokens included
    continuation_tokens: list[str]              # which continuation tokens are present
    prompt_token_weights: list[float]           # normalized cross-attention received by each prompt token
    dominant_prompt_tokens: list[str]           # top-3 prompt tokens by cross-attention received
    anchored_continuation_tokens: list[str]     # cont tokens devoting ≥30% attention to prompt region


class AttentionTrajectory(BaseModel):
    """Joint incremental attention trace as the continuation grows.

    Captures the movement of cross-attention: which prompt tokens remain
    load-bearing as the continuation develops, which fade, and which new
    continuation tokens become semantic pivots.
    """

    checkpoints: list[AttentionCheckpoint]
    prompt_tokens: list[str]                # reference prompt token list
    fading_tokens: list[str]               # high early cross-attention, low late
    persistent_tokens: list[str]           # consistently high cross-attention throughout
    emerging_pivots: list[str]             # prompt tokens gaining cross-attention over time


class TextAttentionAnalysis(BaseModel):
    """Hermeneutic attention analysis: three readings + resonance comparison + trajectory."""

    input_analysis: InputAttentionAnalysis
    continuation_attention: AttentionMap    # clean run on continuation alone
    comparison: HermeneuticComparison
    trajectory: AttentionTrajectory | None = None  # joint incremental trace


# ---------------------------------------------------------------------------
# AttentionAnalyzer
# ---------------------------------------------------------------------------


class AttentionAnalyzer:
    """Bidirectional encoder used as a text-analysis instrument.

    Uses DistilBERT (or a configurable alternative) to analyze the attention
    structure of text.  This is NOT the model under analysis — it is an
    external tool applied to input and output texts as objects.

    The generation mechanism of the model under analysis is never accessed.
    """

    def __init__(self, config: AttentionConfig) -> None:
        # Lazy import — only load transformers components when needed.
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._model = AutoModel.from_pretrained(config.model_name, output_attentions=True)
        self._model.eval()
        self._config = config
        logger.info("Loaded analysis transformer: %s", config.model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_attention(self, text: str) -> tuple[list[str], np.ndarray]:
        """Tokenize text and return ``(tokens, aggregated_attention_matrix)``.

        The attention matrix has shape ``(n_tokens, n_tokens)`` and is
        aggregated across heads and layers according to ``aggregate_method``.
        Special tokens ``[CLS]`` and ``[SEP]`` are excluded from the output.
        """
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_seq_length,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.attentions: tuple of n_layers tensors, each (1, n_heads, seq_len, seq_len)
        attn = torch.stack(list(outputs.attentions))  # (n_layers, 1, n_heads, seq_len, seq_len)
        attn = attn.squeeze(1)  # (n_layers, n_heads, seq_len, seq_len)

        method = self._config.aggregate_method
        if method == "mean_all_layers":
            agg = attn.mean(dim=[0, 1])  # (seq_len, seq_len)
        elif method == "last_layer":
            agg = attn[-1].mean(dim=0)  # (seq_len, seq_len)
        elif method == "mean_upper_half":
            n = attn.shape[0]
            agg = attn[n // 2 :].mean(dim=[0, 1])
        else:
            agg = attn.mean(dim=[0, 1])

        agg_np = agg.numpy()  # (seq_len, seq_len)

        # Get tokens, strip [CLS] and [SEP]
        token_ids = inputs["input_ids"][0].tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(token_ids)

        # Remove special tokens: [CLS] at index 0, [SEP] at index -1
        start, end = 1, len(tokens) - 1
        tokens = tokens[start:end]
        agg_np = agg_np[start:end, start:end]

        # Re-normalise rows to sum to 1 after trimming special tokens
        row_sums = agg_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        agg_np = agg_np / row_sums

        return tokens, agg_np

    def _build_attention_map(self, tokens: list[str], matrix: np.ndarray) -> AttentionMap:
        """Build an :class:`AttentionMap` from tokens and an attention matrix."""
        # Token importance = normalised column sum (how much each token is attended to)
        col_sums = matrix.sum(axis=0)
        col_sums_norm = col_sums / (col_sums.sum() + 1e-10)

        importance = [
            TokenImportance(
                token_str=t,
                token_idx=i,
                importance=float(col_sums_norm[i]),
            )
            for i, t in enumerate(tokens)
        ]

        return AttentionMap(
            tokens=tokens,
            weights=matrix.tolist(),
            token_importance=importance,
            analysis_model=self._config.model_name,
            aggregate_method=self._config.aggregate_method,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_input(
        self,
        prompt: str,
        perturbed_variants: list[str],
    ) -> InputAttentionAnalysis:
        """Analyze prompt structure and how perturbations shift attention.

        A clean independent reading of the prompt as its own text.
        Reveals which tokens are load-bearing without accessing the
        generation model.

        Parameters
        ----------
        prompt:
            The original prompt text.
        perturbed_variants:
            A list of perturbed prompt texts.  At most five are analyzed.

        Returns
        -------
        InputAttentionAnalysis
        """
        tokens, matrix = self._get_attention(prompt)
        attention_map = self._build_attention_map(tokens, matrix)
        base_importance = np.array([ti.importance for ti in attention_map.token_importance])

        deltas: list[AttentionDelta] = []
        for variant in perturbed_variants[:5]:  # cap at 5 variants
            v_tokens, v_matrix = self._get_attention(variant)
            v_map = self._build_attention_map(v_tokens, v_matrix)
            v_importance = np.array([ti.importance for ti in v_map.token_importance])

            # Align by length (take minimum)
            min_len = min(len(base_importance), len(v_importance))
            deltas_arr = v_importance[:min_len] - base_importance[:min_len]

            # Top-5 most affected positions (by absolute delta)
            top_idx = np.argsort(np.abs(deltas_arr))[::-1][:5]
            most_affected = [
                (tokens[i], float(deltas_arr[i]))
                for i in top_idx
                if i < len(tokens)
            ]

            # Find the first position where tokens differ
            changed_idx = 0
            for j, (a, b) in enumerate(zip(tokens, v_tokens)):
                if a != b:
                    changed_idx = j
                    break

            changed_str = v_tokens[changed_idx] if changed_idx < len(v_tokens) else "?"

            deltas.append(
                AttentionDelta(
                    original_token_str=tokens[changed_idx] if changed_idx < len(tokens) else "?",
                    perturbed_token_str=changed_str,
                    token_idx=changed_idx,
                    importance_deltas=deltas_arr.tolist(),
                    most_affected=most_affected,
                )
            )

        return InputAttentionAnalysis(
            prompt_text=prompt,
            attention_map=attention_map,
            perturbation_deltas=deltas,
        )

    def _get_cross_attention(
        self,
        prompt: str,
        continuation_part: str,
    ) -> tuple[list[str], list[str], np.ndarray]:
        """Run attention on [prompt + continuation_part] and extract the cross-block.

        Returns ``(prompt_tokens, cont_tokens, cross_matrix)`` where
        ``cross_matrix[i, j]`` is the attention from continuation token ``i``
        to prompt token ``j``.  The split point is determined by tokenizing
        the prompt alone first.

        Parameters
        ----------
        prompt:
            The original prompt text.
        continuation_part:
            Continuation tokens joined as a string (no trailing space).

        Returns
        -------
        tuple of (prompt_tokens, continuation_tokens, cross_matrix)
        """
        # Count prompt tokens in isolation to find the split point
        prompt_enc = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_seq_length,
        )
        # .shape[-1] works for both (seq_len,) mock shape and (1, seq_len) real HF shape
        n_prompt_with_special = prompt_enc["input_ids"].shape[-1]
        n_prompt = n_prompt_with_special - 2  # strip CLS and SEP

        # Run attention on the joint text
        joint_text = prompt + " " + continuation_part
        all_tokens, all_matrix = self._get_attention(joint_text)

        n_prompt = min(max(n_prompt, 0), len(all_tokens))
        prompt_tokens = all_tokens[:n_prompt]
        cont_tokens = all_tokens[n_prompt:]

        if not prompt_tokens or not cont_tokens:
            n_p = max(len(prompt_tokens), 1)
            n_c = max(len(cont_tokens), 1)
            return prompt_tokens, cont_tokens, np.zeros((n_c, n_p))

        cross = all_matrix[n_prompt:, :n_prompt]  # (n_cont, n_prompt)
        return prompt_tokens, cont_tokens, cross

    def analyze_continuation(self, continuation: str) -> AttentionMap:
        """Clean independent attention run on the generated continuation.

        The continuation is read as its own text — no concatenation with
        the prompt, no joint forward pass.  The result is the continuation's
        own internal attention structure.

        Parameters
        ----------
        continuation:
            The generated continuation text (new tokens only).

        Returns
        -------
        AttentionMap
        """
        tokens, matrix = self._get_attention(continuation)
        return self._build_attention_map(tokens, matrix)

    def compare(
        self,
        prompt_map: AttentionMap,
        continuation_map: AttentionMap,
    ) -> HermeneuticComparison:
        """Analytical resonance comparison of two independent attention readings.

        No joint forward pass.  No concatenation.  The interpretive act is
        explicit: we measure how continuation tokens echo the load-bearing
        structure of the input.

        High resonance score for a continuation token means it directly echoes
        a structurally important prompt token.  Free-floating tokens (resonance
        < 0.2) are continuations that move away from the input's semantic
        anchors.  Anchored tokens (resonance > 0.5) are continuations that
        stay close.

        Parameters
        ----------
        prompt_map:
            AttentionMap from an independent reading of the prompt.
        continuation_map:
            AttentionMap from an independent reading of the continuation.

        Returns
        -------
        HermeneuticComparison
        """
        # Get top-N most important prompt tokens
        prompt_important = sorted(
            prompt_map.token_importance,
            key=lambda t: t.importance,
            reverse=True,
        )[:10]

        prompt_important_strs = {ti.token_str.lower().strip('#') for ti in prompt_important}

        resonances = []
        for cont_ti in continuation_map.token_importance:
            tok = cont_ti.token_str.lower().strip('#')

            # Direct match
            if tok in prompt_important_strs:
                score = 1.0
                anchor = cont_ti.token_str
            else:
                # Weighted similarity: importance-weighted partial match
                best_score = 0.0
                best_anchor = ""
                for p_ti in prompt_important:
                    p_tok = p_ti.token_str.lower().strip('#')
                    # Substring match as a simple proxy for semantic overlap
                    if tok in p_tok or p_tok in tok:
                        s = p_ti.importance * 0.8
                    else:
                        s = 0.0
                    if s > best_score:
                        best_score = s
                        best_anchor = p_ti.token_str
                score = min(1.0, best_score)
                anchor = best_anchor

            resonances.append(TokenResonance(
                token_str=cont_ti.token_str,
                token_idx=cont_ti.token_idx,
                resonance_score=score,
                anchored_to=anchor,
            ))

        mean_resonance = float(np.mean([r.resonance_score for r in resonances])) if resonances else 0.0
        free_floating = [r.token_str for r in resonances if r.resonance_score < 0.2]
        anchored = [r.token_str for r in resonances if r.resonance_score > 0.5]

        return HermeneuticComparison(
            prompt_attention=prompt_map,
            continuation_attention=continuation_map,
            token_resonance=resonances,
            mean_resonance=mean_resonance,
            free_floating_tokens=free_floating,
            anchored_tokens=anchored,
        )

    def analyze_joint_trajectory(
        self,
        prompt: str,
        continuation_token_strs: list[str],
        interval: int | None = None,
    ) -> AttentionTrajectory:
        """Incremental joint attention trace as the continuation grows.

        Runs DistilBERT on ``[prompt + continuation[:k]]`` at regular intervals,
        extracting the cross-attention block to track which prompt tokens remain
        load-bearing and how that grip shifts as the continuation unfolds.

        Parameters
        ----------
        prompt:
            The original prompt text.
        continuation_token_strs:
            Generated token strings in order (e.g. ``OutputStep.selected_token_str``
            for each step).
        interval:
            Checkpoint every N tokens.  Defaults to ``config.trajectory_interval``.

        Returns
        -------
        AttentionTrajectory
        """
        if interval is None:
            interval = self._config.trajectory_interval

        # Baseline prompt token list (reference for all checkpoints)
        prompt_tokens, _ = self._get_attention(prompt)
        n_prompt = len(prompt_tokens)
        n_total = len(continuation_token_strs)

        if n_total == 0:
            return AttentionTrajectory(
                checkpoints=[],
                prompt_tokens=prompt_tokens,
                fading_tokens=[],
                persistent_tokens=[],
                emerging_pivots=[],
            )

        # Build checkpoint steps: every `interval` tokens, always include the final step
        checkpoint_steps = list(range(interval, n_total, interval))
        if n_total not in checkpoint_steps:
            checkpoint_steps.append(n_total)

        checkpoints: list[AttentionCheckpoint] = []
        weight_history: list[list[float]] = []

        for k in checkpoint_steps:
            cont_part = "".join(continuation_token_strs[:k])
            p_toks, c_toks, cross = self._get_cross_attention(prompt, cont_part)

            # Normalized column-sum of cross block: how much cross-attention
            # each prompt token receives from the entire continuation so far
            n_p = n_prompt
            if cross.shape[0] == 0 or cross.shape[1] == 0:
                weights = [0.0] * n_p
            else:
                col_sums = cross.sum(axis=0)  # (n_prompt_actual,)
                total = float(col_sums.sum())
                if total > 0:
                    col_sums = col_sums / total
                # Pad or trim to match baseline prompt token count
                weights = col_sums.tolist()
                weights = (weights + [0.0] * n_p)[:n_p]

            # Top-3 prompt tokens by cross-attention received
            sorted_idx = sorted(range(n_p), key=lambda i: weights[i], reverse=True)
            dominant = [prompt_tokens[i] for i in sorted_idx[:3]]

            # Continuation tokens devoting ≥30% of their attention to the prompt region
            anchored_cont: list[str] = []
            if cross.shape[0] > 0 and cross.shape[1] > 0:
                for ci, tok in enumerate(c_toks):
                    if ci < cross.shape[0] and float(cross[ci, :].sum()) >= 0.3:
                        anchored_cont.append(tok)

            weight_history.append(weights)
            checkpoints.append(AttentionCheckpoint(
                step=k,
                continuation_tokens=list(c_toks),
                prompt_token_weights=weights,
                dominant_prompt_tokens=dominant,
                anchored_continuation_tokens=anchored_cont,
            ))

        # Classify prompt tokens by trajectory pattern across checkpoints
        fading: list[str] = []
        persistent: list[str] = []
        emerging_pivots: list[str] = []

        if len(weight_history) >= 2:
            wh = np.array(weight_history)          # (n_checkpoints, n_prompt)
            n_ck = wh.shape[0]
            early = wh[: n_ck // 2].mean(axis=0)  # first half mean
            late = wh[n_ck // 2 :].mean(axis=0)   # second half mean
            for i, tok in enumerate(prompt_tokens):
                e, l = float(early[i]), float(late[i])
                if e - l > 0.05:
                    fading.append(tok)
                elif l - e > 0.05:
                    emerging_pivots.append(tok)
                elif min(e, l) > 0.08:
                    persistent.append(tok)
        elif weight_history:
            persistent = [
                prompt_tokens[i]
                for i, w in enumerate(weight_history[0])
                if w > 0.1
            ]

        return AttentionTrajectory(
            checkpoints=checkpoints,
            prompt_tokens=prompt_tokens,
            fading_tokens=fading,
            persistent_tokens=persistent,
            emerging_pivots=emerging_pivots,
        )

    def analyze(
        self,
        prompt: str,
        continuation: str,
        perturbed_variants: list[str],
        continuation_token_strs: list[str] | None = None,
    ) -> TextAttentionAnalysis:
        """Run hermeneutic attention: two independent readings + resonance comparison.

        Parameters
        ----------
        prompt:
            The original prompt text.
        continuation:
            The generated continuation text.
        perturbed_variants:
            Perturbed prompt texts for input-side perturbation analysis.
        continuation_token_strs:
            Optional list of individual generated token strings in order.
            When provided, the joint trajectory trace is computed.

        Returns
        -------
        TextAttentionAnalysis
        """
        logger.debug("Hermeneutic attention: reading prompt...")
        input_analysis = self.analyze_input(prompt, perturbed_variants)

        logger.debug("Hermeneutic attention: reading continuation...")
        cont_map = self.analyze_continuation(continuation)

        logger.debug("Hermeneutic attention: comparing structures...")
        comparison = self.compare(input_analysis.attention_map, cont_map)

        trajectory = None
        if continuation_token_strs:
            logger.debug("Hermeneutic attention: joint trajectory trace...")
            trajectory = self.analyze_joint_trajectory(prompt, continuation_token_strs)

        return TextAttentionAnalysis(
            input_analysis=input_analysis,
            continuation_attention=cont_map,
            comparison=comparison,
            trajectory=trajectory,
        )
