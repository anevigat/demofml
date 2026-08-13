# Research Campaign 3 Protocol

Status: active

Effective date: 2026-08-13

This document defines how Campaign 3 research runs are conducted. It binds
every Campaign 3 variant. It does not change any Campaign 1 or Campaign 2
artifact, contract, or result, and it does not weaken the locked-test
prohibition: `2025-01-01` to `2026-03-10` remains forbidden until a Campaign 3
candidate passes the full development acceptance gate.

## Why this protocol exists

The two closed campaigns failed for two different reasons, and only one of them
was a research result.

- **Campaign 1** is a genuine null result. Linear ridge over causal five-minute
  summaries found no cost-aware executable edge across eight pairs and three
  horizons. The engineering was sound; the hypothesis was refuted.
- **Campaign 2** is a methodological failure. Historical results were observed
  before the design was committed, so the periods that could have confirmed the
  design had already been consumed by the decisions that produced it. No amount
  of later computation can restore that independence.

Campaign 1's problem is addressed by testing a different model class (see the
Stage A hypothesis document). Campaign 2's problem is addressed here, by making
the ordering of commitment and observation an auditable property of the
repository rather than an assertion in a document.

## Rule 1 — the sealed envelope

Before a variant's first fold is run, exactly four documents are committed and
their SHA-256 digests are recorded in a sealed-envelope TOML committed in the
same change:

| Role | Artifact |
| --- | --- |
| `hypothesis` | `docs/research/campaign-3-{variant}-hypothesis-v{N}.md` |
| `validation` | the variant's validation contract TOML |
| `model` | the variant's model contract TOML, including the complete hyperparameter search space |
| `acceptance` | the variant's acceptance contract TOML, including every pre-registered threshold |

The envelope is `configs/experiments/campaign-3-{variant}-envelope-v{N}.toml`
and is validated by `demofml.research.envelope`. Its schema is
`sealed-envelope-v1`; it declares exactly these four roles and no others, and
`load_sealed_envelope` rejects any additional role, unknown field, non-UTC
`sealed_at`, or path that escapes the declared repository root.

Verification runs in two places:

- `demofml verify-sealed-envelope --envelope <path>` — on demand, before
  launching a run.
- The development acceptance gate — automatically, whenever the acceptance
  contract declares `sealed_envelope`. A broken seal raises instead of
  producing a failed check, because a gate evaluated against a contract that is
  not the committed one has no interpretation, pass or fail. The resulting
  acceptance report records the envelope id, its `sealed_at` timestamp, its own
  digest, and the digest of all four sealed documents.

The seal proves that the four documents committed before the first fold are
byte-identical to the four documents the gate evaluated. Combined with the
commit timestamps in the repository's history, that is what makes "we decided
this in advance" checkable by someone who does not trust the claim.

## Rule 2 — hyperparameter search stays inside the training window

Campaign 1 never searched hyperparameters at all: `alpha = 1.0` in all three
model contracts of all three lines. Campaign 3 does search, which introduces a
leakage surface Campaign 1 did not have. Therefore:

- The complete search space is declared in the sealed model contract, before
  the first fold. It may not be widened, narrowed, or re-centred afterwards
  without a version bump (Rule 4).
- Selection among candidates occurs **only** within a fold's training window,
  via inner expanding or rolling cross-validation (K = 3-5), with the same
  65-minute purge applied at every inner boundary. The outer validation fold is
  never an input to selection.
- Where a whole-walk-forward comparison of a small fixed candidate set is used
  instead of inner CV, the winner is chosen on the aggregate metric declared in
  the sealed acceptance contract, never fold by fold, and the comparison itself
  is disclosed in the data-use ledger.

## Rule 3 — an observed period is consumed for the decision it informed

A period whose results were observed in order to take a design decision may
still be used for training, but may no longer serve as independent validation
of the decision it informed. This is the rule Campaign 2 broke.

It applies to every observation, not only to formal runs: exploratory screens,
feature-importance inspection, and "quick checks" all consume the period they
touch, for the decision they influence. Gain-based feature importance computed
on training windows is safe to log as a diagnostic; using it to prune features
and re-evaluating on the *same* outer folds is not, and requires a new
versioned variant.

## Rule 4 — post-seal changes require a version bump and a ledger entry

Any change to a sealed document after the first fold has run requires all of:

1. A new version of the changed document (`-v{N+1}`), never an edit in place.
2. A new sealed envelope over the new document set, and a new run id.
3. An append-only entry in `docs/research/campaign-3-{variant}-data-use.md`
   recording the date, what changed, and — critically — **which observed result
   motivated the change**, so a later reader can apply Rule 3 to the new
   variant without reconstructing the history.

The ledger is deliberately not sealed: it is meant to grow, and an envelope
that could absorb it would stop being evidence about what was fixed in advance.

## Naming convention

```
docs/research/campaign-3-{variant}-hypothesis-v{N}.md
docs/research/campaign-3-{variant}-data-use.md
configs/experiments/campaign-3-{variant}-envelope-v{N}.toml
configs/experiments/campaign-3-{variant}-{validation|model|acceptance}-v{N}.toml
```

`{variant}` is `{model-family}-{feature-family}`, for example
`lightgbm-causal-v2`.

## What this protocol does not change

- Contract immutability: a used contract id is never redefined; a changed idea
  gets a new id.
- Atomic, fingerprinted stage outputs and resumable orchestration.
- The acceptance gate's recomputation guarantee: it recomputes metrics from raw
  predictions and replays portfolio accounting instead of trusting any stored
  summary. The seal is an addition to that guarantee, not a substitute for it.
- The locked-test prohibition and its one-shot grant mechanism.
