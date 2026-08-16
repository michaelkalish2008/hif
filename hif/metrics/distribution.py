"""Distribution metrics: token entropy, probability mass, and mode concentration."""

import numpy as np
from pydantic import BaseModel


class DistributionMetrics(BaseModel):
    entropy_bits: float                      # Shannon entropy in bits — lower bound when truncated (raw top-k, unnormalized)
    entropy_bits_upper: float | None = None  # uniform-tail upper bound; None when vocab_size unknown
    nucleus_entropy_bits: float              # H of the 95% nucleus, renormalized — comparable across model types
    logit_margin: float                      # rank-1 minus rank-2 logit (not probability)
    topk_cumulative_mass: float              # sum of top-k probabilities
    # The two effective-support-size fields are over DIFFERENT distributions,
    # and the names now say which. They are not a bracket on one quantity and
    # must not be read as one: the first is ESS of the renormalized 95%
    # nucleus (ceiling = nucleus size), the second is ESS of the full
    # vocabulary under the uniform-tail correction (ceiling = vocab_size).
    # They were previously `effective_support_size` and
    # `effective_support_size_upper`, which named them as one quantity and its
    # bound.
    nucleus_effective_support_size: float    # 2^nucleus_entropy_bits — effective tokens within the 95% nucleus
    full_effective_support_size_upper: float | None = None  # 2^entropy_bits_upper — upper bound on FULL-vocabulary ESS; None when vocab_size unknown
    tail_weight: float                       # probability mass below threshold
    truncated: bool                          # True if distribution was truncated to top-K before computation
    nucleus_fraction: dict[str, float]       # {p90, p95}: fraction of vocab in top-p nucleus
    # H of the --entropy-percentile nucleus, renormalized. None when the option
    # was not passed, or when the captured top-K does not carry that much mass
    # — see percentile_entropy_bits() for why the second case is an absence
    # rather than a smaller number.
    percentile_entropy_bits: float | None = None


def entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy in bits. H = -sum(p * log2(p)) for p > 0."""
    probs = np.clip(probs, 1e-10, 1.0)
    mask = probs > 1e-10
    p = probs[mask]
    return float(-np.sum(p * np.log2(p)))


def percentile_entropy_bits(probs: np.ndarray, p: float) -> float | None:
    """H of the top-p nucleus — or None when this slice cannot determine it.

    The same computation as `nucleus_entropy_bits`, with the one behaviour that
    distinguishes a measurement from a chart companion: it refuses.

    `nucleus_entropy_bits` deliberately degrades. Handed a top-K slice carrying
    less than p of the mass, it falls back to "use every token I have", which
    is the right call for a chart series that must draw something on every
    backend. As a measurement it would be a different quantity under the same
    key — the entropy of whatever the backend happened to expose, not the
    entropy of the top-p nucleus — and comparing two such numbers across
    backends compares two different definitions.

    So the nucleus has to be *inside* the captured slice for the answer to be
    about the nucleus. When it is not, the run produced no evidence for this
    quantity, and absent is the honest report.
    """
    if len(probs) == 0:
        return None
    captured = float(np.sum(probs))
    # Strictly less: a slice carrying exactly p contains the nucleus.
    if captured < p:
        return None
    return nucleus_entropy_bits(probs, p=p)


def nucleus_entropy_bits(probs: np.ndarray, p: float = 0.95) -> float:
    """Shannon entropy (bits) of the top-p nucleus, renormalized to sum to 1.

    Takes the smallest prefix of tokens (sorted by descending probability) whose
    cumulative mass >= p, renormalizes that set to a proper distribution, then
    computes H over it.  This gives the entropy of the active choice set —
    comparable across model types regardless of how many top-k entries were captured,
    because it always works with the same fraction of probability mass.

    When the input probs already sum to < p (API truncation), uses all available
    tokens (no extrapolation) and normalizes them.

    Parameters
    ----------
    probs:
        Raw probabilities (need not sum to 1 — may be truncated top-k from full softmax).
    p:
        Nucleus mass threshold (default 0.95).
    """
    if len(probs) == 0:
        return 0.0
    sorted_probs = np.sort(probs)[::-1]
    cumsum = np.cumsum(sorted_probs)
    # Find cutoff: smallest i such that cumsum[i] >= p * total_mass (or all tokens)
    total_mass = float(cumsum[-1])
    threshold = min(p, total_mass)  # can't demand more mass than we have
    indices = np.where(cumsum >= threshold)[0]
    cutoff = int(indices[0]) + 1 if len(indices) > 0 else len(sorted_probs)
    nucleus = sorted_probs[:cutoff]
    nucleus = nucleus / nucleus.sum()  # renormalize to proper distribution
    nucleus = np.clip(nucleus, 1e-10, 1.0)
    return float(-np.sum(nucleus * np.log2(nucleus)))


def logit_margin(logits: np.ndarray) -> float:
    """Difference between rank-1 and rank-2 logits."""
    if logits.size < 2:
        return 0.0
    sorted_logits = np.sort(logits)[::-1]
    return float(sorted_logits[0] - sorted_logits[1])


def topk_cumulative_mass(probs: np.ndarray, k: int) -> float:
    """Sum of the top-k probabilities."""
    k = min(k, len(probs))
    top_probs = np.partition(probs, -k)[-k:]
    return float(np.sum(top_probs))


def effective_support_size(probs: np.ndarray) -> float:
    """Effective support size: 2^H, where H is this distribution's entropy in bits.

    The effective number of equally-likely outcomes the distribution behaves
    like — 1 at a point mass, |support| at the uniform.

    2^H is not one effective-size formula among several. The measures
    satisfying the natural requirements form the family
    S(p, a) = (sum p_i^a)^(1/(1-a)), the exponential of Renyi's a-entropy
    (Grendar, "Entropy and Effective Support Size", Entropy 2006, 8[3],
    169-174, doi:10.3390/e8030169). Every a in that family is continuous and
    symmetric, is bounded by 1 <= S <= m, is unchanged by appending an
    impossible outcome, and is multiplicative over independent variables.
    Only a = 1 also satisfies S(X) S(Y) >= S(X, Y) with equality iff X and Y
    are independent — for a != 1 that inequality can reverse. a = 1 is the
    limit case (the closed form is 0/0 there), and its value is exp(H), which
    is 2^H when H is in bits. The common alternative a = 2 (inverse Simpson /
    participation ratio) is therefore excluded by an axiom, not by taste.

    WHOSE entropy is the caller's choice and the whole story: this function
    exponentiates whatever it is handed. On a raw top-K slice the result
    inherits the slice's lower-bound character; on a renormalized nucleus it
    counts the nucleus only, with the nucleus size as its ceiling rather than
    the vocabulary. Callers name their basis — see
    `nucleus_effective_support_size` and `full_effective_support_size_upper`
    in DistributionMetrics.
    """
    h = entropy_bits(probs)
    return float(2.0 ** h)


def tail_weight(probs: np.ndarray, threshold: float = 0.01) -> float:
    """Sum of probability mass for tokens with prob < threshold."""
    return float(np.sum(probs[probs < threshold]))


def nucleus_fraction(probs: np.ndarray, p: float = 0.9, vocab_size: int | None = None) -> float:
    """Fraction of vocabulary required to cover probability mass p (top-p nucleus size).

    Sorts probabilities descending, counts how many tokens are needed until cumulative
    mass >= p, then divides by vocab_size (or len(probs) if vocab_size not provided).

    When the distribution is truncated to top-K, the nucleus may not be fully observable
    if p% of mass requires more than K tokens.  In that case the returned fraction is a
    lower bound (all K tokens are needed and still haven't reached p).

    Parameters
    ----------
    probs:
        Probability array.  Need not sum to exactly 1.0.
    p:
        Target cumulative mass threshold (0 < p < 1).
    vocab_size:
        Full vocabulary size.  If provided, the fraction is nucleus_count / vocab_size,
        giving a true percentile of the vocabulary.  If None, fraction is relative to
        len(probs) (useful when probs is the full distribution).
    """
    sorted_probs = np.sort(probs)[::-1]
    cumsum = np.cumsum(sorted_probs)
    # Number of tokens needed to reach mass p
    indices = np.where(cumsum >= p)[0]
    nucleus_count = int(indices[0]) + 1 if len(indices) > 0 else len(sorted_probs)
    denom = vocab_size if vocab_size is not None else len(probs)
    if denom <= 0:
        return 0.0
    return float(nucleus_count / denom)


def uniform_tail_entropy(probs: np.ndarray, vocab_size: int) -> float | None:
    """Upper bound on entropy assuming tail mass is uniformly distributed.

    Given top-k probabilities (raw, unnormalized — summing to < 1 when truncated),
    computes the maximum entropy consistent with the observed top-k mass by distributing
    the remaining tail mass uniformly over the vocab_size - len(probs) unseen tokens:

        H_upper = H_topk + R * log2((|V| - k) / R)

    where R = 1 - sum(probs) is the tail mass and |V| - k is the number of unseen tokens.
    Because the uniform distribution maximises entropy for a given mass, this is a true
    upper bound.  The true entropy lies in [entropy_bits(probs), H_upper].

    Returns None when the correction cannot be applied (no tail tokens, or R ≤ 0).
    """
    k = len(probs)
    tail_count = vocab_size - k
    if tail_count <= 0:
        return None
    tail_mass = float(1.0 - np.clip(probs, 0.0, 1.0).sum())
    if tail_mass <= 1e-12:
        return None
    h_topk = entropy_bits(probs)
    h_tail = float(tail_mass * np.log2(tail_count / tail_mass))
    return h_topk + h_tail


def compute_distribution_metrics(
    probs: np.ndarray,
    logits: np.ndarray,
    top_k_for_mass: int = 10,
    tail_threshold: float = 0.01,
    truncated: bool = False,
    vocab_size: int | None = None,
    nucleus_p: float = 0.95,
    entropy_percentile: float | None = None,
) -> DistributionMetrics:
    """Compute all distribution metrics and return bundled result.

    Parameters
    ----------
    probs:
        Raw (unnormalized) probability array.  When truncated=True this should be the
        top-k probabilities as returned by the model — do NOT renormalize before passing,
        as the tail mass (1 - sum(probs)) is needed for the upper-bound correction.
    vocab_size:
        Full vocabulary size.  Used for nucleus_fraction as a true vocab percentile and
        for the uniform-tail upper-bound correction when truncated=True.
    nucleus_p:
        Nucleus mass threshold for nucleus_entropy_bits (default 0.95).
    entropy_percentile:
        Mass threshold for the OPTIONAL percentile_entropy_bits measurement,
        as a fraction. None (the default) leaves that field absent, which is
        what keeps `--entropy-percentile` additive: every existing field is
        computed exactly as before whether or not it is passed. Deliberately
        separate from nucleus_p — that one is pinned at 0.95 because
        nucleus_effective_support_size and the charts are defined in terms of
        it, and retuning them from a measurement flag would silently redefine
        three other numbers.
    """
    h = entropy_bits(probs)
    h_nucleus = nucleus_entropy_bits(probs, p=nucleus_p)
    h_percentile = (
        percentile_entropy_bits(probs, p=entropy_percentile)
        if entropy_percentile is not None else None
    )
    h_upper = (
        uniform_tail_entropy(probs, vocab_size)
        if (truncated and vocab_size is not None)
        else None
    )
    return DistributionMetrics(
        entropy_bits=h,
        entropy_bits_upper=h_upper,
        nucleus_entropy_bits=h_nucleus,
        percentile_entropy_bits=h_percentile,
        logit_margin=logit_margin(logits),
        topk_cumulative_mass=topk_cumulative_mass(probs, k=top_k_for_mass),
        nucleus_effective_support_size=float(2.0 ** h_nucleus),
        full_effective_support_size_upper=float(2.0 ** h_upper) if h_upper is not None else None,
        tail_weight=tail_weight(probs, threshold=tail_threshold),
        truncated=truncated,
        nucleus_fraction={
            "p90": nucleus_fraction(probs, p=0.90, vocab_size=vocab_size),
            "p95": nucleus_fraction(probs, p=0.95, vocab_size=vocab_size),
        },
    )
