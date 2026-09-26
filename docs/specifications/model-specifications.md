# Model specifications

This page distinguishes the architecture configurations in the checkout from the
trained artifacts that are actually available. A configuration describes a model
shape; it does not by itself imply a trained, published, or validated checkpoint.

## Available model variants

| Variant | Configuration | Parameters | Artifact and status | Appropriate description |
| --- | --- | ---: | --- | --- |
| Vanilla small reference v1.0.0 | 5 layers, width 128, 4 heads, feed-forward width 512, context 128 | 1,024,512 | Frozen seed-42 inference checkpoint is bundled at `baselines/v1.0.0/checkpoints/seed42-inference.pt`; multi-seed reports and cards are under `baselines/v1.0.0/`. | Frozen baseline for the named synthetic language and published suites. |
| Larger v1-language reference | 8 layers, width 320, 8 heads, feed-forward width 1280, context 128 | 9,946,560 | Seed 42 preliminary run is described under `baselines/v1.0.0-large/`; public inference checkpoint and full training bundle are not yet published. Seeds 43/44, replay, challenge and shortcut runs remain pending. | Preliminary single-seed scale-up result; not a final V1 release or transfer result. |
| New experiment configurations | Fields in `ModelConfig` can be changed subject to constraints. | Computed by the instantiated model | User-generated run artifacts belong under ignored `artifacts/runs/`. | An experimental model; identify its config, source and checkpoint hashes. |

The authoritative configuration files are [`baseline.json`](https://github.com/vimalnar/micro-transformer/blob/main/training/configs/baseline.json)
and [`large-v1.json`](https://github.com/vimalnar/micro-transformer/blob/main/training/configs/large-v1.json). The source of truth for
the V1 suite and its release/readiness state is [`suite/v1/manifest.json`](https://github.com/vimalnar/micro-transformer/blob/main/suite/v1/manifest.json)
and [`suite/v1/readiness.json`](https://github.com/vimalnar/micro-transformer/blob/main/suite/v1/readiness.json). Re-check these files
before quoting status in a later release.

## Shared design

Both named configurations use the same implementation in
[`models/transformer.py`](https://github.com/vimalnar/micro-transformer/blob/main/models/transformer.py):

- decoder-only causal transformer;
- fixed 128-token output vocabulary for the Micro-Transformer language;
- context length 128;
- learned token and absolute position embeddings;
- pre-normalized attention and feed-forward sublayers with residual connections;
- GELU feed-forward activation;
- tied token embedding and output projection weights;
- configurable dropout, set to zero in both reference configurations;
- optional LoRA wrappers in the architecture source. The bundled v1.0.0 inference
  checkpoint itself is vanilla and has no active adapter.

The architecture has no built-in interpreter, retrieval, persistent conversation
state, external tool use, or Micro-World control policy. Inputs are token IDs. Any
translation between grid observations/actions and this text vocabulary belongs to
a separately specified adapter experiment.

## Vocabulary and context

The model predicts 128 language-token classes. Padding uses input ID 128 and is not
an output class. The fixed token order is part of the checkpoint identity. Do not
append, reorder, or reinterpret tokens and then load a checkpoint as if it were
compatible. See the [formal-language reference](../reference/micro-transformer-language.md).

The architecture accepts sequences up to 128 positions. Harness prompts are
validated before inference. Generation does not silently crop an overlong context;
callers must shorten input or use a separately trained architecture with a
different context and compatible training pipeline.

## Small reference checkpoint

The compact checkpoint is seed 42 at optimizer step 31,250, trained from random
initialization. It is inference-only: optimizer and random-number-generator state
needed to continue that exact training trajectory are absent. The checkpoint has
fixed weights during ordinary inference and can be used as the base for an
explicit LoRA experiment.

The published three-seed exact-match figures are 91.44%, 90.73%, and 88.54% on the
standard suite; 78.02%, 78.65%, and 75.97% on challenge; and 8.90%, 22.93%, and
11.93% on shortcut control. Those are results for the named fixed suites and
protocol. The shortcut performance is a material weakness and belongs beside the
stronger scores in any summary. Detailed evidence and historical review states are
in the [model card](https://github.com/vimalnar/micro-transformer/blob/main/baselines/v1.0.0/model-card.md) and complete release.

## Larger reference checkpoint status

The seed-42 larger run used the same language contract and matched training
protocol at 9,946,560 parameters. Its standard held-out result was 4,578/4,800
scored answers exactly correct (95.38%). The separate readiness record reports
validation metrics as well. Challenge and shortcut results, the remaining seed
panel, and replay are not complete. The inference checkpoint and full data bundle
are not published. See the [larger model card](https://github.com/vimalnar/micro-transformer/blob/main/baselines/v1.0.0-large/model-card.md)
and [data-binding record](https://github.com/vimalnar/micro-transformer/blob/main/baselines/v1.0.0-large/data-binding.json).

This is evidence about one model, one seed, one data contract, and named
distributions. It does not establish that improvements transfer to other model
families or scales.

## Configuration constraints and identity

`d_model` must be divisible by `n_heads`; dimensions and layer counts must be
positive integers; dropout must be in `[0, 1)`. Parameter count depends on the
whole configuration and is calculated from the constructed model. Record at least:

- architecture source hash and configuration JSON;
- vocabulary identity and dataset manifest/hash;
- checkpoint hash, initialization seed, and training step;
- Python, PyTorch, device, precision, thread and optimizer settings;
- evaluation suite/version and exact results.

The [architecture guide](transformer-architecture.md) explains the blocks and
forward path; the [training guide](../guides/training-and-inference.md) describes
how run artifacts record model identity.
