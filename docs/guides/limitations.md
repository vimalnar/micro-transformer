# Limitations and claim boundaries

This page summarizes what the implemented system and published results do—and do
not—support. Read it with the relevant model card, dataset card and experiment
report; those provide artifact-specific evidence.

## Formal-language model

- The vocabulary is fixed at 128 tokens, grammar is controlled, and context is
  limited to 128 tokens in the named reference models.
- The bundled checkpoint answers one self-contained prompt at a time. It has no
  conversation memory, retrieval, external tools or online learning.
- It can make errors on valid prompts. The harness exposes named successes and
  preserved failures; these cases are examples, not a representative estimate.
- Published accuracy is tied to identified synthetic suites and protocols. The
  shortcut control is weak for the small reference.
- A passing reference interpreter check establishes the label under that
  interpreter. It does not establish that a neural model used the intended
  mechanism to produce its answer.

## Larger model

The 9,946,560-parameter model has one completed seed-42 run and a preliminary
standard held-out result. Challenge and shortcut suites, the remaining seed panel,
replay, and public checkpoint/data release remain pending in the recorded
readiness state. Do not describe this as a final multi-seed release or evidence
that the effect transfers to larger or unrelated models.

## Micro-World

The grid simulation and formal-language interpreter are separate systems. The
language checkpoint was not trained as a world-action policy. An adapter interface
defines a possible experiment, but does not itself demonstrate perception,
planning, robust control or task completion. The browser inspector's full map is
for human inspection and is not the agent's partial observation.

The optional HTTP/WebSocket service stores sessions in process memory. The
repository does not provide production authentication, persistent sessions,
multi-process coordination or deployment hardening.

## Evidence and interpretation

Task names such as state tracking, delayed recall, belief, communication, spatial
relations and temporal reasoning refer to benchmark constructions. Correct answers
on them do not by themselves establish the corresponding human capacity or a
general reasoning faculty. A negative result can reflect the model, representation,
adapter, task distribution or another implementation dependency; report plausible
alternatives rather than attributing cause without a design that distinguishes
them.

Execution, conformance, scientific validity, review approval and outcome are
separate states. Preserve failed or inconclusive runs and all relevant controls.
See the [experiment report specification](../specifications/experiment-report-v1.md)
and [reproducibility guide](reproducibility.md).

## Consciousness and welfare

This project does not establish subjective experience, consciousness, sentience,
selfhood or moral patienthood in any current model. Fluent language, benchmark
performance, functional capability and architectural labels are insufficient on
their own. The broader humanistic research aspiration should remain distinct from
empirical claims about the systems documented here.
