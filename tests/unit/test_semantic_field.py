"""Tests for the within-generation semantic field instrument, Veer (◈).

The Veer rule (docs/ARCHITECTURE.md § Field-model notes): Veer is the
per-step displacement of the top-K
candidate cloud's semantic centroid; deformation is the per-step change in the
cloud's spread. Uses a content-addressable (deterministic-per-string) embedder so
identical candidate clouds → identical centroids → zero Veer.
"""

import hashlib

import numpy as np

from hif.analysis.semantic_field import SemanticFieldAnalyzer
from hif.hourglass.output_side import OutputSideTrace
from hif.models.base import StepRecord, TopKEntry


class HashEmbedder:
    """Deterministic per-string unit vectors (same string → same vector)."""

    def __init__(self, dim: int = 16):
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).random(self._dim)
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.array(out, dtype=np.float64)


def _step(idx, cands, sel=None):
    topk = [
        TopKEntry(token_id=i, token_str=s, prob=float(p),
                  logit=float(np.log(p)), logprob=float(np.log(p)))
        for i, (s, p) in enumerate(cands)
    ]
    sel_str = sel if sel is not None else cands[0][0]
    return StepRecord(step=idx, selected_token_id=0, selected_token_str=sel_str, topk=topk)


def _trace(steps):
    return OutputSideTrace(
        steps=steps, input_ids=[1], generated_ids=[0] * len(steps),
        prompt_text="p", model_name="m", top_k=4, max_new_tokens=len(steps),
        seed=0, mean_step_entropy=1.0,
    )


def test_none_below_two_steps():
    a = SemanticFieldAnalyzer(HashEmbedder())
    assert a.analyze(_trace([_step(0, [("x", 1.0)])])) is None


def test_identical_clouds_zero_veer():
    # context_window=0 → bare token strings; identical clouds every step.
    cloud = [("cat", 0.5), ("dog", 0.3), ("fish", 0.2)]
    steps = [_step(i, cloud, sel="cat") for i in range(4)]
    r = SemanticFieldAnalyzer(HashEmbedder(), context_window=0).analyze(_trace(steps))
    assert r is not None
    assert len(r.veer) == 3 and len(r.deformation) == 3
    assert r.n_steps == 4
    assert r.max_veer < 1e-9          # identical centroids → no movement
    assert r.mean_deformation < 1e-9  # identical spread → no reshape


def test_changing_clouds_positive_veer():
    # Each step draws from a distinct vocabulary → the semantic centre moves.
    vocabs = [
        [("apple", 0.6), ("pear", 0.4)],
        [("river", 0.6), ("lake", 0.4)],
        [("engine", 0.6), ("piston", 0.4)],
        [("verse", 0.6), ("rhyme", 0.4)],
    ]
    steps = [_step(i, v, sel=v[0][0]) for i, v in enumerate(vocabs)]
    r = SemanticFieldAnalyzer(HashEmbedder(), context_window=0).analyze(_trace(steps))
    assert r is not None
    assert len(r.veer) == 3
    assert all(0.0 <= x <= 2.0 for x in r.veer)
    assert r.mean_veer > 0.0 and r.max_veer >= r.mean_veer
    assert r.n_steps == 4


def test_empty_topk_steps_are_skipped():
    # Two consecutive defined steps then a trailing empty step: the (0→1) pair is
    # defined; the empty step contributes no centroid and no veer pair.
    steps = [
        _step(0, [("a", 1.0)]),
        _step(1, [("b", 1.0)]),
        StepRecord(step=2, selected_token_id=0, selected_token_str="", topk=[]),
    ]
    r = SemanticFieldAnalyzer(HashEmbedder(), context_window=0).analyze(_trace(steps))
    assert r is not None
    assert r.n_steps == 2        # two defined centroids
    assert len(r.veer) == 1      # only the (0→1) pair


def test_no_consecutive_defined_pair_returns_none():
    # A defined step separated from another by an empty step → no veer pair → None.
    steps = [
        _step(0, [("a", 1.0)]),
        StepRecord(step=1, selected_token_id=0, selected_token_str="", topk=[]),
        _step(2, [("b", 1.0)]),
    ]
    assert SemanticFieldAnalyzer(HashEmbedder(), context_window=0).analyze(_trace(steps)) is None
