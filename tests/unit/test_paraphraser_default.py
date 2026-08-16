"""The configured paraphraser and the generator's own default must agree.

`PerturbationConfig.paraphraser_model` is a literal so config.py need not
import the perturbation package. That is a duplicated constant, and a
duplicated constant drifts — this is the check that it has not.

Naming it matters beyond tidiness: the paraphraser writes the text that
`perturbation_jsd_bits`, both input-entropy rows and `io_cosine_similarity`
are computed against, so it is part of the instrument. A record saying only
"local" would not identify which instrument produced the numbers.
"""

from hif.config import PerturbationConfig
from hif.perturbation.local_llm import DEFAULT_LOCAL_MODEL


def test_config_default_matches_the_generator_default():
    assert PerturbationConfig().paraphraser_model == DEFAULT_LOCAL_MODEL


def test_the_shipped_default_is_the_local_paraphraser():
    # Read from conftest, which pins the SUITE to "rule" for speed and
    # hermeticity — PerturbationConfig() here would report the test pin, not
    # what a real run does.
    from tests.conftest import SHIPPED_PARAPHRASER

    assert SHIPPED_PARAPHRASER == "local"


def test_the_record_names_which_checkpoint_wrote_the_variants():
    assert PerturbationConfig().paraphraser_model, (
        "a run must record which checkpoint wrote its variants; "
        "'local' alone does not identify an instrument"
    )
