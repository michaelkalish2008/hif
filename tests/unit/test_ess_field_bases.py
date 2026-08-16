"""The two ESS fields are over different distributions, and the names say so.

`effective_support_size` was 2^H over the renormalized 95% nucleus while
`effective_support_size_upper` was 2^H over the full tail-corrected vocabulary
— named as one quantity and its bound, computed over two different bases.
These pin the bases to the names.

ESS itself is not a measurement and must not become one: 2^H is a bijection of
entropy, so it discloses nothing entropy does not and fails the Significance
Gate's distinct-disclosure condition.
"""

import numpy as np
import pytest

from hif.metrics.distribution import (
    compute_distribution_metrics,
    nucleus_entropy_bits,
)
from hif.profile.registry import MEASUREMENT_KEYS

# A genuine top-K slice: sums to < 1, so there is tail mass for the
# uniform-tail bound to be defined over.
TRUNCATED_SLICE = np.array([0.35, 0.22, 0.13, 0.09, 0.06, 0.04])


def _metrics(vocab_size=1000, truncated=True):
    return compute_distribution_metrics(
        TRUNCATED_SLICE, np.log(TRUNCATED_SLICE),
        vocab_size=vocab_size, truncated=truncated,
    )


def test_no_ess_key_in_the_measurement_set():
    """The gate rejects it: 2^H discloses nothing H does not."""
    for key in MEASUREMENT_KEYS:
        assert "effective_support" not in key


def test_nucleus_field_exponentiates_the_nucleus_entropy():
    result = _metrics()
    assert result.nucleus_effective_support_size == pytest.approx(
        2.0 ** nucleus_entropy_bits(TRUNCATED_SLICE, p=0.95)
    )
    # ...and NOT the raw top-K entropy, which is the confusion the rename
    # exists to prevent.
    assert result.nucleus_effective_support_size != pytest.approx(
        2.0 ** result.entropy_bits
    )


def test_upper_field_exponentiates_the_full_vocabulary_bound():
    result = _metrics()
    assert result.entropy_bits_upper is not None
    assert result.full_effective_support_size_upper == pytest.approx(
        2.0 ** result.entropy_bits_upper
    )


def test_the_two_fields_are_not_a_bracket_on_one_quantity():
    """Different bases, different ceilings — not an interval on one number."""
    result = _metrics()
    assert result.full_effective_support_size_upper > result.nucleus_effective_support_size


def test_upper_field_absent_without_a_vocab_size():
    p = np.array([0.5, 0.3, 0.2])
    result = compute_distribution_metrics(p, np.log(p))
    assert result.full_effective_support_size_upper is None
