# Release status and versioning

This page distinguishes available artifacts from configurations and preliminary
results. Re-check the linked manifests when preparing a report or downstream use;
status may change as releases are completed.

## Small language baseline v1.0.0

The repository includes the frozen seed-42 inference checkpoint and model/dataset
cards for the small vanilla baseline. The cards report a three-seed evaluation on
the standard, challenge and shortcut suites. The bundled checkpoint contains the
weights required for inference, but not the optimizer and random state required to
continue its exact original training trajectory.

The full reproducibility release is separate from normal Git history. Consult
[`baselines/v1.0.0/README.md`](https://github.com/vimalnar/micro-transformer/blob/main/baselines/v1.0.0/README.md) for the current
release-artifact instructions and
[`suite/v1/manifest.json`](https://github.com/vimalnar/micro-transformer/blob/main/suite/v1/manifest.json) for the machine-readable
suite identity.

## Larger v1-language reference

The larger reference has one completed seed-42 run and a preliminary standard
held-out result. The release manifest marks the model as `preliminary_single_seed`.
Seeds 43 and 44, a seed-42 replay, challenge and shortcut evaluation, and public
checkpoint/data publication are pending in the current artifact record. It should
not be presented as a final V1 scaling release.

The configuration and evidence are under [baselines/v1.0.0-large](https://github.com/vimalnar/micro-transformer/tree/main/baselines/v1.0.0-large).
That directory's README, model card, manifest and data-binding file are the
authoritative details for artifact availability.

## New experiments

An experiment configuration is not a release. Identify any new checkpoint by its
architecture/configuration, vocabulary, data, seed, training endpoint, runtime and
evaluation suite. Do not reuse a version label or describe a local result as
published unless the corresponding artifact and evidence are actually available.

The project does not promise a stable hosted inference API or production service.
The included CLI, Python modules, HTTP routes and data formats should be treated
according to their versioned contracts and the specific release notes, not as an
enterprise support commitment.
