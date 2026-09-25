# Preliminary small-versus-large comparison

Status: diagnostic single-seed comparison. This is not a final scale-transfer
claim.

| Model | Parameters | Standard exact answer accuracy | Evaluation questions |
|---|---:|---:|---:|
| v1.0.0 small, seed 42 | 1,024,512 | 91.44% | 4,800 |
| larger v1-language, seed 42 | 9,946,560 | 95.38% | 4,800 |

The small-model value is the published v1.0.0 seed-42 reference result. The
larger value comes from the completed seed-42 endpoint and the temporary frozen
inference harness run. Both use the standard held-out distribution. The larger
model also reached 95.29% exact answer match on its 5,000-record validation
endpoint and 95.72% answer-macro accuracy.

## Interpretation

The preliminary result is consistent with a useful capacity improvement on this
distribution, but it does not isolate model size as the only cause. The larger
run uses the current repository training entry point while the archived small
baseline was produced by the historical native training wrapper, although the
declared data, architecture source, objective, optimizer settings, and endpoint
protocol were matched. A final comparison must complete the seed panel, verify
the implementation/environment identity, and evaluate challenge and shortcut
suites before stronger conclusions are made.

## Not tested here

- seeds 43 and 44;
- independent seed-42 replay;
- challenge and shortcut held-out suites;
- transfer to Micro-World control;
- general reasoning, persistent memory, agency, or consciousness.
