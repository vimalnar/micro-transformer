# Choose a path through the project

Micro-Transformer is a small, inspectable research and teaching baseline. The
right way to use it depends on whether you want to learn the components, study a
controlled model question, or assess a product idea. The repository does not
provide a hosted production model platform.

## Hobbyists and students

Start with the [setup guide](setup.md), then use the frozen checkpoint through the
[harness](harness.md). Try the [worked examples](examples.md), inspect the
[formal language](../reference/micro-transformer-language.md), and read the
[architecture walkthrough](../specifications/transformer-architecture.md) before
changing code.

Good learning exercises include tracing one token through the model, changing one
synthetic fact and deriving the expected answer, exploring an explicitly documented
failure, or changing a single architecture component in a local experiment. Keep
the model prediction separate from the reference interpreter's answer.

The small vocabulary and synthetic tasks make the system easier to inspect. They
also sharply limit what its performance says about ordinary language or real-world
tasks.

## Academic researchers

Start by choosing which system is under study:

- formal-language data or interpreter;
- transformer architecture or training objective;
- Micro-World simulation or agent observation;
- a separately implemented adapter connecting a model and the world.

Write the hypothesis, intervention, comparison, controls, endpoint, stopping rule
and claim boundary before running the experiment. Use the
[experiment guide](experiment-guide.md), [report specification](../specifications/experiment-report-v1.md)
and [reproducibility checklist](reproducibility.md). For an architecture claim,
use matched from-scratch runs across the declared seeds. Preserve all seeds and
task breakdowns. Use protected evaluation only after development decisions are
fixed.

The formal-language baseline has frozen multi-seed evaluation, but the shortcut
control is weak. The larger reference is preliminary single-seed evidence with
challenge/shortcut evaluation and the remaining seed panel outstanding. Micro-World
language-model control is not established by the existence of an adapter or a
feasibility controller.

## Product and company R&D teams

Use the repository to make an early experiment concrete: define a target behaviour,
build a small task with explicit truth and failure cases, compare a minimal model
change under controls, and decide whether a larger evaluation is justified. Record
compute cost, latency where measured, error modes, data assumptions and what must
be retested in the intended deployment model.

This is an evaluation/prototyping workbench. It does not include production
deployment, access control, tenant isolation, audit/compliance tooling, a maintained
hosted service, scalable inference, or a general model governance stack. The local
Micro-World service is in-memory and intended for interactive research use. A
successful small-scale result is a reason to design a scale-appropriate test, not
a guarantee of transfer or production performance.

## Claims all audiences should avoid

The benchmark's words for task templates (including belief, memory, communication
and planning) name formal input/output patterns. They do not establish human-like
mental capacities. Fluent output, performance, architecture names and functional
similarity alone do not show consciousness or subjective experience. See
[limitations and claim boundaries](limitations.md).
