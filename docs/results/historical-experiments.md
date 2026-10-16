# Historical experiments: shelved agent and living-world work

**Status: historical, shelved work. These experiments are not part of the current
supported Micro-Transformer baseline.**

On 15 September 2026, local agent and living-world development was shelved. The
local Git stash records this boundary as `shelve agency and living-world work
2026-09-15` (stash commit `2c29d92`). That stash is a local recovery record, not a
published release or a dependency of the documented setup.

## What the retained records contain

The external report collection includes exploratory agent training, recovery,
rehearsal, transfer, temporal and readout variants, and living-world trials.
Representative directory names include `agent-bootstrap-f2-suite-01`,
`agent-action-recovery-core-standard-01`, `agent-access-replay-transfer-01`, and
`organism-frozen-01`. These names identify historical runs; they do not establish
successful capabilities.

For example, the manifest for `agent-action-recovery-core-standard-01` identifies
a checkpoint by hash, development-evaluation seeds, candidate and control arms,
and a source inventory. That inventory references files such as
`training/train_agent.py` and `scripts/run_organism_agent.py`, which are absent from
the current baseline source tree. Running these experiments requires their exact
historical implementation and artifacts, with separate dependency and provenance
checks; the current quickstart does not reproduce them.

The documentation inventory did not independently validate every historical
report, checkpoint, source archive or scientific conclusion. File presence and
experiment names are not evidence of protocol conformance, successful transfer,
agency or consciousness. Preserve negative and inconclusive results when reviewing
these records.

## Where the records are stored

In the maintainer's local storage layout, reports were moved to an external volume
to free disk space:

```text
/Volumes/EVERYTHING BACKUP/Development/micro-transformer/artifacts/reports/
```

This is a local storage location, not a public download or a path that other users
must create. See [external artifact storage](../guides/reproducibility.md#external-artifact-storage)
for how to locate records without changing their recorded identities.

## Relationship to current results

The current suite is identified by `suite/v1/manifest.json` and
`suite/v1/readiness.json`. Its bounded Micro-World reference, `REF-MW-001`, is a
separate experiment with a retained negative frozen-model result. The historical
agent reports neither replace that result nor complete the outstanding larger
reference-model evaluation and publication work.

See [release status](../guides/release-status.md) and the
[V1 readiness audit](v1-readiness-audit.md) for the supported baseline and its
remaining limitations. Resuming historical work requires an explicit new scope
and evaluation plan; documenting it does not reactivate it.
