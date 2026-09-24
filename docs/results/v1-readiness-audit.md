# V1 readiness audit, excluding the deferred larger model

Date: 24 September 2026.
Decision: **The seven requested phases pass the pre-V1 gate.** The toolkit is
usable for bounded research workflows, but the final V1 claim remains blocked by
the intentionally deferred larger reference model.

## Requirement audit

| Requirement | Evidence | Finding |
|---|---|---|
| Frozen V1 contracts | `suite/v1/manifest.json`; contract verifier | Pass: canonical paths, versions, hashes, and the deferred blocker are machine-readable |
| One canonical harness | `harness/`; parity and self-tests | Pass: it imports the repository model and checkpoint rather than copied model artifacts |
| Data generation | generator v4.1.0; full verifier | Pass: fresh train, validation, and test splits are generated with declared seeds and disjointness checks |
| Model source, config, and weights | model source, baseline config, v1.0.0 manifest and checkpoint | Pass for the small reference model; its complete local release verifies 179 files |
| Clean Python interoperability | public compatibility environment and test suite | Pass on Python 3.12/macOS arm64 with public PyTorch 2.8.0 and NumPy 2.0.2; all 90 tests pass outside the repository working directory |
| Training and held-out evaluation | full verifier | Pass: prepare, train from scratch, save, reload, and evaluate run as one clean smoke path |
| Benchmark and report specification | `benchmarks/v1/manifest.json`; experiment report specification; comparison tool | Pass: suites, splits, controls, metrics, endpoints, states, and claim boundaries are explicit |
| Micro-World | six-family benchmark and `REF-MW-001` | Pass as a bounded environment and reference workflow: scripted feasibility is 12/12 and all replays match |
| Reference experiment | `benchmarks/v1/reference-results.json`; hash-checked raw decisions | Pass with a negative result: the frozen model is 0/12 with 0% complete-command parsing under the untrained adapter |
| Documentation and repository readiness | README, quick start, contribution guide, citation metadata, GitHub Actions gate | Pass locally; the workflow is configured but has not run remotely, and publication remains a separate release operation |
| Larger reference model | explicitly deferred | Not implemented; this is the sole intentionally deferred technical V1 artifact and blocks final V1 |

## Audience usefulness

### Hobbyists and students

The repository now provides one inspectable route from a fixed language and data
generator through training, evaluation, and interactive frozen inference. A user
can run a short verification path, modify a copied architecture, and see both
working examples and known failures without needing a hosted service or framework.

### Academic researchers

The suite is useful for bounded, reproducible comparisons because data/model
identity, controls, seeds, endpoints, raw run outputs, validity state, negative
results, and limitations are separated. It does not by itself establish external
validity, statistical power, or transfer beyond the declared tasks. New claims
still require a suitable seed policy, fresh protected evaluation, and review.

### Product R&D teams

The suite can cheaply test whether a model-level or model-to-environment idea is
coherent enough to justify a larger experiment. The Python packages import cleanly
outside the checkout, and the Micro-World loop records executable actions and
physical outcomes. It is not a production serving, security, governance, RAG, or
agent platform, and small-model results must be retested at product scale.

## Residual risks and release work

- Produce the larger reference model with source, config, weights, seed policy,
  held-out results, and matched small-model comparison before declaring final V1.
- Publish or otherwise make the complete three-seed baseline release retrievable;
  its manifest currently has no public download URL.
- Run the compatibility gate on Linux and Windows if those platforms are to be
  advertised as verified rather than merely supported by the Python packaging.
- Keep the final Micro-World cases as public regressions. Any new scientific
  claim needs fresh protected cases rather than tuning against `REF-MW-001`.
- Publishing the repository, preprint, or website remains a separate release
  decision; this audit makes no claim that those external actions occurred.

## Acceptance decision

The requested seven phases are complete when the commands in the quick start pass
from a clean checkout with the documented public dependencies. The only permitted
suite status at this point is `v1_ready_except_larger_reference_model`. Changing it
to final V1 requires the deferred model and a new audit.
