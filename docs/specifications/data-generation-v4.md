# Micro-Transformer data generation v4

## Purpose and boundaries

Generate reproducible, interpreter-labelled training and evaluation episodes for
the vanilla Micro-Transformer baseline. Vocabulary remains the original **128 tokens**
with exactly the same token IDs. This changes task coverage, not the transformer,
gridworld, or language semantics. `.` ends statements, `?` ends a question, and `|`
ends an independent episode. A record has zero or one question.

The default generator now uses 13 registered families and 51 variants. Composition,
state changes, distraction and answer balance provide difficulty; adding arbitrary
vocabulary is not necessary. No finite generator covers every future capability.
New capabilities should be explicit, versioned additions with their own tests.

## Task coverage

| Family | Variants and important cases |
|---|---|
| Statement only | Executable events with no question |
| State tracking | One or repeated moves; take/drop; swaps; containment/removal; unknown location |
| Delayed recall | Updated target location amid intervening events, including swaps and take/drop |
| Ownership | A single take; transfers; drop with/without retaking; relocation clearing ownership; unowned objects |
| Property | Colour updates; positive/negated attributes; same/different colours; unknown colour |
| Conditional | Positive and negated conditions; independently varied condition truth and branch effects |
| Belief | Current, stale, updated and missing observations |
| Spatial | True/false behind, beside, with and contains relations; which/unknown responses |
| Quantifier | Counts including zero; true/false some, none and all |
| Temporal | Before, after, and, while, again, only and because; order-sensitive same-target actions |
| Communication | Current, stale, updated, missing and explicitly supplied information |
| Ability | Can/cannot open; visible/not visible after state changes |
| Composition | Mixed moves, painting, taking, dropping and giving, followed by location, owner or colour questions |

Statements use the existing reference interpreter. Its conventions are deliberately
limited: `where` reports an explicit location or `unknown`; `while` has deterministic
left-to-right execution; `because` does not introduce a general causal solver;
belief/knowledge stores are explicit symbolic records. These task names do not
imply capabilities beyond their tested definitions.

Examples of valid training content include:

```text
ava move cube two to garden. ben move cube two to hall. where cube two? hall |
ava move cube two to garden. ben take cube two. who has cube two? ben |
ben take cube two. ben drop cube two at hall. who has cube two? unknown |
ava lock door one. ava unlock door one. can ben open door one? yes |
ava unlock door one. ava lock door one. can ben open door one? no |
ava move cube two to garden. is cube two heavy? no |
ava move cube two to garden. is cube two not heavy? yes |
```

These illustrate semantics; the generator additionally enforces each output
split's permitted object identities.

## Generate, validate, prepare

Run from the project root. Generation uses Python's standard library only.

```bash
python3 scripts/generate_dataset.py \
  --episodes 500000 --split train --seed 46090 \
  --output artifacts/datasets/my-train.jsonl \
  --require-full-coverage --require-capability-coverage

# Optional: put approximately 10% of the fixed 500,000 records into pairs.
python3 scripts/generate_dataset.py \
  --episodes 500000 --split train --seed 46090 \
  --paired-counterfactual-fraction 0.10 \
  --output artifacts/datasets/my-paired-train.jsonl

python3 scripts/generate_dataset.py \
  --validate artifacts/datasets/my-train.jsonl \
  --require-capability-coverage

.venv-training/bin/python -m training.prepare_data \
  --input artifacts/datasets/my-train.jsonl \
  --output artifacts/datasets/prepared/my-train
```

Use new paths for a new dataset. Existing outputs are refused unless explicitly
replaced with `--overwrite`. Failed generation does not publish an incomplete
corpus over the previous dataset. `--resume` continues an unfinished generation
with the same configuration and source identity; it is not for appending a
different profile or changing an old completed dataset.

Validate independent corpora separately. Multiple paths supplied to one
`--validate` command are treated as contiguous shards of the same corpus, with
continuous record IDs and one aggregate coverage check.

For larger requests, change `--episodes` to the required count and optionally add
`--shard-size 100000`. The number requested is **episodes**, not tokens. Episode
lengths are variable and the manifest reports their total and distribution.
`--max-tokens 128` bounds episode length; `0` removes the limit. Generation rejects
over-length candidates rather than silently truncating their meaning.

Output remains JSONL with token strings and answer labels, plus non-input metadata:

```json
{"schema":"micro-transformer-jsonl-v4","id":0,"tokens":["ava","move","cube","five","to","hall",".","where","cube","five","?","hall","|"],"answer_tokens":["hall"],"task":"state_tracking","variant":"moves","task_version":"1.0.0","suite":"standard","difficulty":"mixed","statement_count":1,"structural_template":"state_tracking:moves","split":"train","split_key":"cube:five"}
```

The trainer receives token IDs only. Task, variant, suite and expected-answer
metadata are not extra model features. Prepared format v2 retains variant metadata
for evaluation; readers still accept existing prepared v1 datasets and checkpoints.

## Optional paired counterfactual records

`--paired-counterfactual-fraction F` enables controlled two-record pairs for
`0.0 <= F <= 1.0`; it defaults to `0.0`. The requested episode count remains the
final record count. The generator chooses the nearest feasible number of pairs as
`min(floor(episodes / 2), floor(episodes * F / 2 + 0.5))`, so paired records are
always even and an odd-sized fully paired request retains one unpaired record.

Both members begin with the same structured scenario. A `flip` pair changes one
task-relevant state field and must change the interpreter-derived answer. An
`invariant` pair changes the location of one fresh distractor object and must keep
the answer. Each member is replayed independently by the reference interpreter,
and dataset validation additionally checks pair completeness, metadata agreement,
answer relation, split agreement, unique IDs, and the declared one-field surface
difference. Pair rows add `pair_id`, `pair_member`, `pair_relation`,
`intervention_field`, and `intervention_role`; these fields are not encoded by
`training.prepare_data` and do not reach the model. Records are deterministically
shuffled and counterparts are kept non-adjacent whenever the dataset size permits.

Invariant pairs cover every queried built-in family and variant. Flip pairs cover
state tracking, delayed recall, ownership, colour properties, conditionals,
belief, non-containment spatial relations, counting, temporal tasks,
communication, opening ability, and all composition variants. Flip pairing is
excluded for generic positive/negated attributes (not every attribute can be
cleared in the language), containment (false requires an extra removal event),
`some`/`none`/`all` (one move cannot guarantee every polarity), and visibility
(the two polarities require unequal hide/find histories). Statement-only records
have no supervised target. Extension tasks are left unpaired until an explicit
structured intervention contract is added.

The manifest reports requested and realized fractions, pair and paired-record
counts, relation/task/variant counts, exclusions, paired and unpaired target
balances, adjacency diagnostics, and semantic integrity results. Pair-enabled
runs currently cannot use `--resume`; start them afresh with `--overwrite` after
an interruption. Ordinary generation retains its existing resume behavior and,
with the option omitted or zero, its byte-identical seeded JSONL output.

Paired counterfactuals are a training-data intervention intended to emphasize
input/output dependencies. They are not memory, retrieval, recurrence, persistent
state, or evidence of any special higher-level capability.

Sharded JSONL can be prepared one shard at a time. The current trainer consumes
one prepared directory per run; multi-shard sampling is not implemented. For the
current 500k corpus, a single prepared directory is sufficient.

## Reproducibility and measured coverage

The manifest records the generator/schema versions, source hash, extension-module
hashes, vocabulary IDs, seed, profile, suite, exact task quotas, variant counts,
answer counts by variant, event features, lengths, output hashes, and rejection
counts. Resume checks configuration identity, including exclusion-file hashes.

`--require-full-coverage` checks token coverage. `--require-capability-coverage`
checks that every selected task variant appears and that each declared boolean
variant contains both `yes` and `no`, differing in count by no more than one.
These are distinct checks. Small requests may legitimately fail either gate.

Task percentages form a shuffled, prefix-stable 100-record cycle. Variants cycle
within each task; boolean outcomes alternate within each declared boolean variant.
Random entities, actors, locations and histories vary independently. Duplicate
rejection does not advance the variant or change its intended boolean outcome.
Counters are reconstructed when resuming.

Each builder executes its events to derive the answer. Validation then parses and
replays the serialized episode from a fresh state. **This uses the same reference
interpreter implementation**, not a second independent semantics engine. Handwritten
golden cases and analytical tests separately check key semantics. This avoids
claiming that interpreter bugs are impossible.

The current implementation streams output, but duplicate fingerprints and the task
schedule use memory proportional to episode count. Millions of episodes are
supported by the interface; memory and runtime still grow with the request. It is
not a constant-memory, unlimited-throughput generator.

## Evaluation partitions

Use distinct seeds and `--split train`, `validation`, or `test`. The existing
kind/identifier partition remains disjoint: 205/26/25 entities respectively.
Standard held-out data uses the same families but new identities and event samples.

This partition also limits numeric coverage: the current test/validation split has
only one or two available identifiers per kind. Counting uses up to five objects
in training but cannot test that full range within this held-out identity scheme.
A broader counting experiment needs a separately versioned split/task protocol.

`--suite challenge` requests longer histories in builders that use update counts:
five to seven rather than the default mixed difficulty's one to four. It is a
stress distribution, **not a guarantee that every challenge structure is unseen**.
For example, the moves variants expose a direct longer-chain holdout; variants
without configurable update chains are not length-held-out. Select and report
variants explicitly when making a structural generalization claim.

```bash
python3 scripts/generate_dataset.py \
  --episodes 3000 --split test --suite challenge --seed 46093 \
  --output artifacts/datasets/my-challenge.jsonl \
  --disjoint-from artifacts/datasets/my-test.jsonl \
  --require-full-coverage --require-capability-coverage
```

The exclusion file prevents exact overlap with the standard test. It does not
alone establish unseen semantics. Use validation for model selection, keep final
tests out of training, and retain failures in per-variant inference reports.

## Adding a future capability

Core infrastructure and task content are separated:

```text
scripts/generate_dataset.py       Command-line entry point
micro-world/micro_transformer/data/generator.py   Vocabulary, interpreter, validation and dataset lifecycle
micro-world/micro_transformer/data/task_registry.py  Task registration and extension loading
micro-world/micro_transformer/data/task_families.py  Built-in scenario builders
micro-world/micro_transformer/data/coverage.py    Coverage and answer-balance measurements
```

Legacy v3 helper builders remain for compatibility with direct imports. The CLI
uses the registered v4 families; new work should use the registry, not add another
entry to the legacy `BUILDERS` mapping.

For a capability expressible with the existing language:

1. Create a local Python module exporting `register(registry, api)`.
2. Register a uniquely named `TaskSpec`: version, description, variants, builder,
   and any boolean variants requiring balance.
3. The builder accepts `(Context, occurrence)` and returns an `Episode`. Reuse
   `Scenario` to execute events and derive labels. Returned task/version/variant/suite
   metadata must match the contract. `Scenario.finish()` defaults to task version
   `1.0.0`; set `episode.task_version` explicitly for a different version.
4. Supply a profile JSON mapping task names to positive integer percentages
   summing to 100. A profile can mix built-in and extension tasks.
5. Add golden cases, positive/negative examples, difficult compositions and a
   held-out protocol. Test generation, validation, preparation and model consumption.

The checked-in example is executable:

```bash
python3 scripts/generate_dataset.py \
  --episodes 1000 --split train --seed 17 \
  --task-module scripts/examples/location_check_task.py \
  --profile-file scripts/examples/location_check_profile.json \
  --output artifacts/datasets/custom-task.jsonl \
  --require-capability-coverage
```

The example demonstrates a custom containment check. Use `--list-tasks` with the
same module flag to inspect its registered contract. Modules execute Python code:
load only trusted local files. Extension modules are not loaded by the trainer;
ordinary validation/preparation checks their serialized language and labels.
To certify their declared variant coverage using `--validate`, supply the extension
and profile flags too.

A new **meaning or vocabulary token** is a larger change: version the language,
update the interpreter/reference and tokenizer, add semantic tests, and adapt
model input/output dimensions where needed. It is not silently introduced by
registering a new task name.

## Current corpus and compatibility

The original v3 and v4 experiments are documented in `docs/results/`. Generated
corpora, checkpoints and prepared data are local artifacts rather than repository
source. Git history is the canonical source-code record; manifests retain generator
versions, source hashes and output hashes for locally generated datasets.
The initial v4 manifests received a documented wording correction in v4.0.1:
fresh replay was incorrectly described as a separate independent interpreter.
Episode bytes and original producer hashes were preserved.

The generator, corpus and model should be versioned together in experiment reports.
Do not infer full language competence from token coverage or a high aggregate
accuracy alone.
