# Contributing a measurement

There is one contribution path in this project: **adding a measurement**. It
is deliberately small — five steps, two files, one registry row — and this
document is the whole of it. (Its subject is declared in the same row: a
quantity that turns out to be about the prompt rather than about the model is
still welcome, it just lands in a different block of the record. Step 2.) (Bug fixes are always welcome and need no
ceremony; everything below is about growing the measurement set without
growing its confusion.)

A measurement here is a **triple** — *observable × functional × resolution* —
reported as a run-level scalar in its natural unit. Read the front matter of
[docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) first; it defines each coordinate
and the rules the steps below enforce.

One meta-rule, stated once because it is exactly the kind of staleness a new
contributor introduces by accident: **do not write the number of measurements
anywhere** — not in docs, comments, help text, or tests. The registry is the
count; `hif schema` reports it. Anything the registry can answer, prose must
derive, phrase around, or point at the command for.

---

## The five steps

### 1. Compute the quantity, in natural units, in the right module

Put the computation where its inputs live: perturbation-response quantities in
`hif/metrics/stability.py`, per-step distribution quantities in
`hif/metrics/distribution.py`, step-to-step output divergence in
`hif/metrics/shift.py`, semantic/embedding quantities in
`hif/metrics/semantic.py` or an analyzer under `hif/analysis/`, prompt-side
quantities in `hif/hourglass/input_side.py`. The function returns the value —
or `None`.

**Never in a chart module.** `hif/viz/` draws numbers; it does not define them.
A quantity computed inside `hif/viz/signals/*.py` is reachable only by someone
looking at a chart, and the CLI cannot report it — which is how Shift ◆ came to
be visible on the companion website and impossible to reproduce with `hif
profile`. The rule the fix established: the computation lives under
`hif/metrics/` (or `hif/analysis/`) and *both* the chart and `measurements()`
import it, so a reader's number and a record's number cannot drift apart. If
you are about to write arithmetic in a viz module, that is the signal you are
adding a measurement and should be here at step 1.

Three hard requirements. Each was learned the expensive way, and each has a
reason of one line:

- **Never normalise an unbounded quantity.** The normaliser leaks into the
  number: a `/ log₂(vocab_size)` denominator once surfaced as the strongest
  apparent "behavioural" feature in the study corpus (r = 0.980, constant
  within a model) — tokenizer metadata masquerading as behaviour — and bounded
  scales saturate exactly where resolution matters.
- **Never invert into a score.** `1 − x` hides the measurement behind a score:
  the reader sees "stability 1.0" where the instrument measured "shift 0.0
  bits", and the ceiling pins there while the underlying quantity still moves.
- **Absent, not pinned.** When the run produced no evidence for the quantity
  (backend can't teacher-force, an optional stage didn't run, too few
  variants), return `None` and let the key be **omitted** from the record. A
  fabricated `0.0` or `1.0` is a measurement claim; "no evidence" is not.
- **Absent when the computation stops being the quantity.** The rule above
  extends past "could not measure" to "measured something else". If a backend
  degrades your inputs until the arithmetic answers a different question, the
  key must be omitted, not emitted with a comment. Worked case: on a
  selected-only backend the per-step distributions are point masses, so a JSD
  between them is `0` when the tokens agree and `1` bit when they differ — a
  token-disagreement rate. `perturbation_jsd_bits` and `output_step_jsd_bits`
  go absent there. If you think the degraded quantity is worth reporting, it
  needs its own registry row with a key that names what it actually is, and it
  must pass the Significance Gate on its own; overloading the original key is
  the failure the `subject` field was introduced to stop.
- **A truncation-limited quantity is not the same thing.** A number that is
  still the quantity, but bounded in a stated direction by top-K truncation, is
  *reported* with the bound named in its `definition` — `output_entropy_bits`
  is a lower bound and says so. Where the bound is large enough to need
  quantifying, ship the bound as its own measurement rather than as a caveat
  flag: `output_step_topk_overlap_fraction` is the resolution limit on
  `output_step_jsd_bits`, and each definition names the other. A flag is an
  adornment a consumer must know to look for; a registry row is a second fact.

### 2. Declare its triple — and its subject

Decide, before writing the registry row:

- **observable** — what it is computed from (input distribution, output
  distribution, attention row, embeddings of text…).
- **functional** — one of `FUNCTIONALS` in `hif/profile/signals.py`:
  `information-theoretic` (entropy, surprisal, JSD, trace correlation) or
  `geometric` (embedding distance, cluster structure).
- **resolution** — one of `RESOLUTIONS`: `per-step` or `per-position` when the
  scalar summarises a within-run trace at that granularity, `aggregate` when
  the quantity exists only at whole-run level. Do not invent a new resolution
  value for a quantity that fits an existing one.
- **subject** — one of `SUBJECTS`: *whose behaviour the number describes*. The
  triple says what was measured; the subject says who it is about, and the two
  are not the same question. See the Subject section of
  [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) for the enum and the principle.

Answer the subject question by asking what moves the number, not what runs in
the pipeline:

- Does it come from the target's own forward pass? → `target-distribution`.
- Does a fixed local instrument (embedder, analysis encoder, teacher-forcing
  surrogate) read text the *target actually produced*? → `target-output-text`.
  A proxy on the target's real data is a reading instrument, and legitimate.
- Does it couple a target-derived series with one derived from something else?
  → `mixed`. Do not lump it with either side; a correlation cannot be
  attributed to one of its two series.
- Can the number be computed without the target running at all? → `prompt-only`.

Then declare **`subject_under_surrogate`** if a surrogate can change the
answer: it is what `subject` becomes when the surrogate named by the row's
`surrogate_group` stands in, and `None` when the subject does not change.
Subject is backend-dependent for most input-side and output-side rows, and
must be modelled that way rather than pinned to one value that is wrong on
half the backends.

**A measurement whose subject is `prompt-only` does not go in the measurement
set.** It is emitted in the record's `prompt_measurements` block, alongside
the reference model that produced it — `measurements()` filters it out
automatically once the row declares the subject, so the only thing you must
get right is the declaration. A prompt-only quantity is still worth having:
it is comparable across targets precisely because the target does not enter
it. It is simply not a fact about the target, and the record must not be able
to imply that it is. A caveat flag is not a substitute for absence here — that
was the exact failure this field exists to prevent, and the prompt-only
measurements in the predecessor audit showed zero variance across every
model-side change tested, which is what "cannot see the model" looks like in
data.

### 3. Check it passes the Significance Gate

Both conditions, from docs/MEASUREMENTS.md — this is the acceptance bar:

1. **Derivability** — computable from the distributional observable alone, no
   inference to hidden structure.
2. **Distinct disclosure** — it must disclose a facet no admitted measurement
   already captures, and move independently of them somewhere across contexts.

The second condition rejects most candidates. Precedents: `continuity` was
`1 − sensitivity` from the same JS divergences; the `wager` aggregate was
byte-for-byte the `surprise` aggregate; ESS is entropy in different units.
Each of those was removed, and a new measurement that fails the same test will
not be admitted. If your quantity is an existing one re-scaled, re-signed, or
re-averaged, it is not a new measurement.

### 4. Register it — one row

Add one `Measurement(...)` row to `MEASUREMENT_REGISTRY` in
`hif/profile/signals.py`, and emit the value from `measurements()` in the same
file (guarded so absence omits the key). The registry row is the single
extension point: the CLI table, `hif schema`, the Markdown reports, `compare`,
the record path, and the backend capability guard (`hif/models/capabilities.py`)
all derive from it — there is no second list to update. In particular, adding a
measurement requires touching `capabilities.py` **zero times**: its capability
sets are comprehensions over the registry rows (`surrogate_group`,
`observable`, `needs_distribution_pair`), and
`tests/unit/test_capability_sets.py` fails if a row and the sets ever disagree.
Declare what your measurement needs on the row itself — a `surrogate_group`,
and `needs_distribution_pair=True` if it is computed from a divergence between
two per-step token distributions — and the guard follows.

Row conventions:

- **`key`** — descriptive and unit-suffixed (`*_bits`, `*_fraction`,
  `*_cosine`, `*_r`). The key is the stable machine name; it never changes
  after release.
- **`label`** — only an *established* shorthand from the docs ("Wager ▲",
  "Continuity"). If the quantity has no canonical shorthand, leave `None`; a
  made-up name is worse than none.
- **`definition`** — one or two sentences: what it is, its bound (or that it
  is unbounded), and when it is absent.
- **`surrogate_group`** — `"input"` or `"output"` if the value can come from a
  teacher-forcing surrogate on restricted backends, else `""`. This is a claim
  about the computation, so check it against `hif/profile/builder.py` rather
  than against intuition: three rows had it wrong before the subject field
  forced the audit — one flagged a proxy that never ran, two consumed the
  proxy basis without declaring it.
- **`subject`** and **`subject_under_surrogate`** — from step 2. A row that
  declares `subject_under_surrogate` must also declare a `surrogate_group`,
  or the degradation can never fire; the registry invariants enforce this.

### 5. Add a test

Two kinds, both cheap:

- **A value test** for the computation, in the module's existing test file,
  following the existing pattern — construct small synthetic inputs where the
  answer is checkable by hand, and assert the natural-unit property. See
  `tests/unit/test_sensitivity_stability.py::test_measurements_are_in_natural_units`,
  which asserts `input_entropy_shift_bits == mean(|1|, |6|, |1|) = 8/3` and
  `input_entropy_std_bits == std([1, 6, 1], ddof=1)` — values above 1.0, proving
  nothing squashed them into `[0, 1]` — and an absence test asserting `None`
  (not `0.0`) when the evidence is missing.
- **The registry invariants** in `tests/unit/test_measurement_registry.py`
  cover your row automatically (key uniqueness, complete row, valid triple,
  valid subject, a declared degradation that can actually fire, and
  emitted-implies-registered). You do not need to touch them — just run them.
  They also assert that `measurements()` and `prompt_measurements()` partition
  the run's values, so a prompt-only row lands in the right block by
  construction.

Then run the whole suite: `.venv/bin/python -m pytest`.

---

## Worked example: `input_entropy_std_bits`

The most recently added measurement (hif-v2.1, with
`branch_pairwise_cosine_similarity`), and a model of the diff shape: it
touched **two files** — the module that computes it and the registry that
declares it.

**The quantity.** The perturbation stage already measured
`input_entropy_shift_bits`, the *mean* per-variant input entropy shift. The
*spread* of those shifts across variants — does every paraphrase move the
model's input entropy by the same amount, or do some barely move it while one
moves it a lot? — is a different disclosure: two runs with the same mean shift
and very different spreads are behaving differently, and no admitted
measurement saw it. Gate passed: derivable from the same teacher-forced input
distributions (condition 1), moves independently of the mean (condition 2).

**File 1 — `hif/metrics/stability.py`** (compute, in natural units, absent
when unmeasurable):

```python
# Spread of the entropy response. Needs >= 2 variants to mean anything.
input_entropy_std_bits: float | None = None
if n_in >= 2:
    input_entropy_std_bits = float(np.std(entropy_shifts, ddof=1))
```

Raw bits — no normaliser, no inversion — and `None` (not `0.0`) for a single
variant, because one shift has no spread: that is "no evidence", not "measured
zero". The value rides on the existing `PerturbationResponse` model as an
optional field.

**File 2 — `hif/profile/signals.py`** (declare the triple and emit the value):

```python
Measurement(
    key="input_entropy_std_bits",
    name="Input entropy shift spread (bits)",
    unit="bits",
    definition=(
        "standard deviation (ddof=1) of the per-variant input entropy "
        "shifts. ... Unbounded above; absent when fewer than two "
        "variants exist."
    ),
    observable="input distribution",
    functional="information-theoretic",
    resolution="aggregate",
    subject=SUBJECT_TARGET_DISTRIBUTION,
    subject_under_surrogate=SUBJECT_PROMPT_ONLY,
    label="Stability",
    surrogate_group="input",
),
```

and in `measurements()`:

```python
if getattr(st, "input_entropy_std_bits", None) is not None:
    out["input_entropy_std_bits"] = st.input_entropy_std_bits
```

That is the whole diff, plus the test assertion shown in step 5 and — because
it grew the measurement set — a minor bump of `SIGNAL_SET_VERSION`
(`hif-v2` → `hif-v2.1`; minor versions are additive supersets, so `hif
compare` still works across them). Everything else — the CLI table row, the
`hif schema` entry, the Markdown report rows, comparability in `compare` —
derived from the registry row with no further edits.

**Documentation:** add a per-measurement section to docs/MEASUREMENTS.md
Part 1 (and Part 2 if the resolution is `per-step`/`per-position` with a
surfaced trace), following the shape of the existing sections: Zone,
Definition, Unit and range, Absent-when. No counts, anywhere.
