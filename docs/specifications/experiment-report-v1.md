# Experiment and report specification v1

Every reference or contributed experiment must distinguish what was built, what
ran, whether it conformed to the declared protocol, whether the evidence is valid
for the question, and what scientific outcome was observed.

## Required identity

- Experiment ID and title.
- Hypothesis stated before evaluation.
- Baseline and intervention.
- Source revision and architecture hash.
- Model/checkpoint, dataset, benchmark, world and adapter versions.
- Training and evaluation seeds.
- Runtime, device, dependency versions and elapsed compute.

## Required design

- Independent variable and exact change from baseline.
- Controls and ablations.
- Development, validation and final-test boundaries.
- Primary and secondary endpoints.
- Stopping rule and action/compute budget.
- Known confounds and information available to each system.

## Required evidence

- Raw predictions or decisions.
- Per-seed and pooled measurements.
- Failures as well as successes.
- Hashes for input data, raw outputs and released artifacts.
- Reproduction commands.

## Separate states

Reports must record these separately:

1. `execution`: whether the run completed technically.
2. `conformance`: whether it followed the declared contract.
3. `validity`: whether the evidence can answer the stated question.
4. `review`: review status, if any.
5. `scientific_outcome`: positive, negative, inconclusive or not tested.

A completed run is not automatically valid, and a valid negative result is not a
failure of execution.

## Claim boundary

Conclusions apply only to the named model, data, task, controls, seeds and
interfaces. A narrower diagnostic must not be presented as confirmation of a
broader hypothesis. Micro-scale behaviour does not establish larger-scale
transfer, general intelligence, agency, subjective experience or consciousness.
