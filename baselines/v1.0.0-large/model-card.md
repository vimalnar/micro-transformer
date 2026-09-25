# Micro-Transformer larger v1-language model card

Status: preliminary single-seed reference, not a final V1 release.

## Summary

This model is a from-scratch scale-up of the small v1 reference model. It keeps
the ordered 128-token language, dataset contract, objective, optimizer settings,
and evaluation interface fixed while increasing model capacity to 9,946,560
parameters.

## Architecture

The model uses `models/transformer.py` with 8 layers, width 320, 8 attention
heads, feed-forward width 1280, context length 128, bias enabled, and no dropout.
It has no pretrained weights, retrieval, external memory, simulator connection,
or rule-based answer path.

## Training

Seed 42 was trained from scratch for 31,250 optimizer steps over the verified
packed one-million-record v1 training release. Training used CPU float32, four
threads, batch size 32, AdamW with learning rate 0.0003, weight decay 0.01,
epsilon 1e-8, `foreach=False`, gradient clipping 1.0, no scheduler, and
deterministic algorithms. The complete seed panel has not yet been run.

## Preliminary evaluation

On the 5,000-record standard held-out suite, the endpoint generated 4,578 of
4,800 scored answers exactly (95.38%). The harness verified that all outputs
terminated, weights remained unchanged, and repeated greedy inference was
deterministic. Challenge and shortcut suites have not yet been evaluated for
this model.

## Limitations and intended use

This is a research artifact for controlled, inspectable experiments. It should
not be described as a general language model, a reasoning system, an agent, or
evidence of transfer from small to larger models. Task labels such as belief,
memory, and communication name benchmark templates; they do not establish those
human capacities. Composition and property tasks remain materially weaker than
several other task families.

The public checkpoint and complete packed-data release are not yet uploaded.
`data-binding.json` therefore records a local prepared release with no public
download URL.
