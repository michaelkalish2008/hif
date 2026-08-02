# Contributing a measurement

There is one contribution path in this project: **adding a measurement**. It
is deliberately small — five steps, two files, one registry row — and this
document is the whole of it. (Bug fixes are always welcome and need no
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
`hif/metrics/distribution.py`, semantic/embedding quantities in
`hif/metrics/semantic.py` or an analyzer under `hif/analysis/`, prompt-side
quantities in `hif/hourglass/input_side.py`. The function returns the value —
or `None`.

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

### 2. Declare its triple

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
and the record path all derive from it — there is no second list to update.

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
  teacher-forcing surrogate on restricted backends, else `""`.

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
  emitted-implies-registered). You do not need to touch them — just run them.

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
