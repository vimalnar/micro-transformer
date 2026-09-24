# Micro-Transformer V1 suite: gap and execution plan

Status: seven implementation phases and their pre-V1 audit completed locally on
24 September 2026. The larger reference
model is explicitly deferred by the project owner and remains required before a
complete V1 claim. This plan keeps Micro-Transformer a small, inspectable research
toolkit. It does not add a new orchestration CLI, native LangChain/LangGraph
support, visualization platform, MicroVision, or MicroAudio.

## Implementation status

| Phase | Evidence | Status |
|---|---|---|
| V1 contracts and source of truth | `suite/v1/manifest.json`, `scripts/verify_v1_suite.py` | Implemented |
| Repository consolidation | canonical `harness/` using repository model and checkpoint; historical checkout parity test | Implemented; historical repository preserved |
| Data and Python interoperability | full generate -> prepare -> train -> reload -> evaluate smoke verification | Implemented and passed locally |
| Benchmark and report contract | `benchmarks/v1/manifest.json`, report specification, report comparison tool | Implemented |
| Minimal Micro-World reference | six scenario families, four controls, frozen-model adapter, raw decisions and replay | Implemented; final reference outcome negative |
| Reference packaging and documentation | quick start, experiment guide, result report, contribution and citation files | Implemented |
| Pre-V1 gate | contract, full-pipeline, clean-install, reference-result and complete test matrix | Passed; see `docs/results/v1-readiness-audit.md` |
| Larger reference model | model/config/weights/matched comparison | Deferred; blocks final V1 |

## V1 outcome

V1 is complete when an academic researcher, hobbyist, or product R&D team can:

1. define or select a controlled task;
2. generate and validate data;
3. train or modify a model;
4. run a frozen baseline or variant;
5. evaluate it with held-out tests and controls;
6. optionally execute a model through a documented Micro-World adapter;
7. record the configuration, identities, results, limitations, and reproduction
   steps in a report another person can rerun.

“Complete” means a complete research suite and proving ground, not a general AI
platform or evidence that small-model behaviour transfers automatically to larger
models.

## Existing inventory

### Micro-Transformer repository

- **Language:** the 128-token language, vocabulary order, grammar, semantics, and
  interpreter are documented in `docs/reference/` and implemented in
  `micro-world/micro_transformer/data/`.
- **Data:** generator v4.1.0 supports deterministic JSONL generation, task
  families, entity-disjoint splits, coverage checks, validation, manifests,
  sharding, and optional paired counterfactuals. Prepared-data conversion is in
  `training/prepare_data.py`.
- **Model:** `models/transformer.py` exports the small configurable causal
  transformer. `training/configs/baseline.json` defines the canonical 5-layer,
  width-128, 4-head, 512-FF, context-128 model with 1,024,512 parameters.
- **Baseline:** `baselines/v1.0.0/` contains the seed-42 inference checkpoint,
  cards, manifest, sample data, results, and a link to the locally prepared full
  release. The full release contains three seeds, packed evaluation data,
  predictions, reports, and verification evidence.
- **Training/evaluation:** `training/train.py`, `runtime.py`, `harness.py`,
  `infer.py`, `evaluate.py`, metrics, checkpoint snapshots, resume, copied
  architectures, and LoRA are implemented and covered by tests.
- **Micro-World:** `micro-world/micro_world/` contains a headless 16x16 simulator,
  typed observations/actions/events, replay records, rendering/audio cues, a
  FastAPI service, browser inspector, versioned model adapter, and a six-family
  closed-loop benchmark. Its contract is documented in
  `docs/specifications/gridworld-v1.md`.
- **Experiment guidance:** `docs/guides/experiment-guide.md` and
  `docs/guides/micro-world-model-adapter.md` describe comparison discipline and a
  minimal adapter contract. Baseline reports and the complete-release evidence
  are already present locally.

### Integrated and historical harnesses

- `harness/talk_to_micro_transformer.py` is now the canonical human-facing entry
  point and loads the repository's model source and checkpoint directly.
- The historical `micro-transformer-harness` checkout remains untouched as
  rollback/release evidence. A regression test confirms its copied model artifacts
  still match the canonical hashes. It is no longer the development source of
  truth.

### Micro-World

- The simulator remains independently testable and is joined to the language model
  only through the versioned V1 adapter. `REF-MW-001` is now a first-class bounded
  reference experiment with feasibility, wait, random and reactive controls. Its
  frozen-model result is negative and retained.

## Mapping to the 10 critical V1 artifacts

| # | Critical artifact | Current state | Classification | Minimum closure work |
|---|---|---|---|---|
| 1 | MicroLanguage-128 specification | Canonical specification, ordered-vocabulary hash and runtime compatibility checks | Complete for pre-V1 | Preserve version and hash whenever language-dependent artifacts change |
| 2 | MicroTransformer-128 baseline | Source, config, seed-42 weights, cards, hashes, results and verified local three-seed release | Complete for pre-V1 | Publish the complete release without changing its recorded identity |
| 3 | Larger reference model | Explicitly deferred by the project owner | Critical for final V1 | Add source, config, weights, declared seed policy and matched held-out comparison; do not imply transfer |
| 4 | Data generator | Canonical v4.1.0 entry point, schema, split contract, disjointness and full smoke workflow | Complete for pre-V1 | Preserve old release data provenance rather than assuming generator equivalence |
| 5 | Micro-World V1 | Headless simulator plus versioned adapter, six scenario families, controls, hash-checked raw decisions and replay | Complete for pre-V1 | Use fresh protected cases for new claims |
| 6 | Training scripts | Prepare, train, resume, infer, evaluate and LoRA paths; clean public dependency path verified | Complete for pre-V1 | Add the larger-model recipe during the deferred phase |
| 7 | Evaluation/benchmark suite | Unified manifest covers language and Micro-World suites, controls, splits, metrics and claim boundaries | Complete for pre-V1 | Expand only in response to a defined research question |
| 8 | Experiment + report specification | Canonical report contract and explicit-metric comparison tool | Complete for pre-V1 | Require it for published reference experiments |
| 9 | Reference experiments | Small-model baseline and bounded Micro-World adapter experiment complete; small-vs-larger comparison deferred | Partial by explicit deferral | Produce the matched larger-model experiment in the deferred phase |
| 10 | Documentation + GitHub release | Suite map, quick start, guides, citation and contribution metadata complete locally | Technically ready; publication pending | Publish repository/release only after owner review and larger-model completion if labelled final V1 |

## Cross-cutting gaps and risks

### Critical for final V1

- **Larger reference artifact:** the configurable architecture is not a substitute
  for a named, trained, evaluated and reproducible larger model. This is the sole
  intentionally deferred technical blocker.
- **Public release retrieval:** the complete three-seed small-model release verifies
  locally, but its manifest does not yet provide a public download URL.

### Important after V1

- Expand Micro-World scenario families and task difficulty after the first
  end-to-end contract is stable.
- Add more systematic larger-model transfer targets and provider-specific adapters.
- Improve package ergonomics, tutorials, and example notebooks without creating a
  second source of truth.
- Decide whether the standalone harness should be generated from, or checked
  against, the main repository release; retain it as a slim public artifact either
  way.
- Add directly verified Linux and Windows environments before advertising them as
  verified platforms.

### Later

- New visualization products or a platform-level inspector.
- Native LangChain/LangGraph integrations.
- MicroVision, MicroAudio, persistent services, plugin registries, or a larger
  orchestration CLI.
- Claims about general reasoning, agency, consciousness, or automatic transfer.

## Ordered implementation phases

### 1. Freeze V1 contracts and source of truth

Implemented through the machine-readable suite manifest and verifier. Canonical
paths, versions and the deferred larger-model obligation are explicit.

### 2. Consolidate the harness

Implemented under `harness/`. The wrapper uses the canonical architecture and
checkpoint. The historical checkout remains preserved until a later explicit
archive decision.

### 3. Close data and interoperability seams

Implemented through vocabulary/generator/baseline contract checks and a full
fresh-data smoke path covering generation, disjoint splits, preparation, training,
checkpoint reload, held-out evaluation and harness verification.

### 4. Define benchmark and report contracts

Implemented through the V1 benchmark manifest, experiment-report specification,
non-overwriting report writers and declared-metric comparison tool.

### 5. Implement the bounded Micro-World reference experiment

Implemented with six scenarios: direct navigation, detour navigation, key/door
access, object pickup and delivery, pressure-plate causality, and visit-and-return
memory. Feasibility, wait, uniform-random and reactive-no-history controls are
included. The frozen-model validation result is negative and retained.

### 6. Package reference workflows and documentation

Implemented through the integrated README, audience-specific quick start,
experiment guide, Micro-World adapter guide, reference result, contribution guide,
citation file and reproducibility commands.

### 7. Run the pre-V1 gate

Passed locally. The gate verifies all contracts, the full interoperability path,
all tests, reference reports, clean installation, and audience-usefulness evidence.
Its status is `v1_ready_except_larger_reference_model`; see
`suite/v1/readiness.json` and `docs/results/v1-readiness-audit.md`.

## Deferred phase: larger reference model

After the seven phases pass their audit, choose the smallest clearly larger model
that remains practical on ordinary hardware. Train the declared seed set from
scratch, select only on validation, evaluate on the same untouched suites, and
preserve every seed and failure. Source, config, checkpoint, run manifest and model
card must remain together. This phase is required before changing the suite status
to final V1.

Pre-V1 acceptance gate:

- every non-larger-model artifact has an authoritative path and verifier;
- benchmark, report specification and end-to-end Micro-World diagnostic are
  reproducible;
- source, data, configs, weights, reports and controls are linked;
- clean Python interoperability is verified on supported environments;
- the larger model is clearly recorded as the sole intentionally deferred V1
  artifact;
- claims remain limited to measured tasks and interfaces.

## Decision rule for future additions

An addition belongs in V1 only if it closes one of the ten artifacts, reduces a
reproducibility/integration risk, or is required by one of the two reference
experiments. Otherwise record it as Important after V1 or Later. This keeps the
suite small enough to inspect while making its negative and inconclusive results
as usable as its positive results.
