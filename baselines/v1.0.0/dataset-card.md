# Dataset card: corrected controlled baseline corpus

## Composition

- 1,000,000 training episodes: 960,000 questions and 40,000 statements.
- 250,000 paired records forming 125,000 complete pairs: 62,500 answer-flip pairs
  and 62,500 answer-invariant pairs.
- 5,000 validation and 5,000 standard held-out episodes, plus separate challenge
  and shortcut-control suites in the complete release.
- Thirteen families including statement-only serialization; twelve queried
  families and 51 queried variants.
- A fixed 128-token vocabulary and maximum episode length of 128 tokens.

The queried families cover state tracking, delayed recall, ownership, properties,
conditionals, beliefs, communication, spatial relations, quantifiers, temporal
constructions, abilities, and composition. Data contains direct questions,
updates, superseded facts, distractors, negative answers, and unknown answers.

## Generation and truth

Episodes are generated procedurally with fixed seeds. The deterministic reference
interpreter produces the answer and validates semantics. Interpreter results and
scorer annotations do not enter the transformer's runtime input. Pair identities,
dependency positions, task labels, expected relations, and audit fields are
scorer-only.

`sample-training-data.jsonl` contains short human-inspectable sequences adapted
from actual corpus records. Distractors were removed for readability, so the file
is illustrative rather than a byte-exact corpus subset, statistically representative
sample, or training input.

## Reproduction boundary

The complete release includes the exact packed model-facing train, validation, and
standard arrays. Those arrays reproduce every input and next-token target but not
the full scorer-only annotations from the original multi-gigabyte JSONL. The data
generator can create new deterministic corpora; a newly generated corpus must not
be described as byte-identical to the archived training corpus unless every
record and manifest hash matches.
