# Worked examples

The bundled harness includes a few verified examples and known failures. A larger
set of hobby examples is maintained in the companion
[micro-transformer-examples repository](https://github.com/vimalnar/micro-transformer-examples).
Those cases use the canonical frozen checkpoint in this toolkit checkout. They are
small, authored demonstrations, not a held-out benchmark.

## Try one question

With the repository environment installed, run:

```bash
.venv-training/bin/python harness/talk_to_micro_transformer.py
```

At the prompt, enter a complete episode using known vocabulary:

```text
ava move coin three to garden. where coin three?
```

This asks for the last stated location of `coin three`. The model generates a
completion; it does not invoke the reference interpreter to answer the prompt.

Other vocabulary-valid patterns include:

```text
ava paint coin three red. what colour coin three?
ava take coin three. who has coin three?
ava tell ben coin three at garden. where ben knows coin three?
```

Examples are syntactically illustrative. A valid sentence is not a guarantee that
the model learned its intended answer. Start with `:examples` for predictions
verified against the exact bundled checkpoint.

## Run the companion case collection

Clone `vimalnar/micro-transformer-examples` separately. Its runner accepts the
path to this toolkit checkout, for example:

```bash
python3 run_examples.py \
  --harness-dir /path/to/micro-transformer \
  --device cpu
```

Use `--only communication,conditional` to select cases. Each authored worked
example checks labels with the toolkit's v1 reference interpreter, then gives only
the prompt to the frozen model. It writes a report with expected answer, observed
prediction, artifact identities and a frozen-weight check. Choose a new report
directory for each run; existing outputs are protected from overwrite.

The companion collection currently groups cases around communication, conditional
state, ownership, counting and two retained diagnostic failures (colour update and
swap followed by relocation). See that repository's README for its current run
options and exact case inventory.

## Design a useful variation

1. Change one controlled part of a prompt: agent, object, location, distractor, or
   delay between fact and question.
2. Confirm that every word and construction belongs to the language contract.
3. Derive the expected answer independently with the reference interpreter.
4. Run the unchanged frozen checkpoint and record its actual completion.
5. Keep every mismatch. Do not edit the label to fit the model output.
6. If you repeatedly use cases for development, do not present them as untouched
   final evaluation.

These steps support an inspectable teaching exercise. They do not establish
general reasoning, durable memory, communication ability beyond the named cases,
agency, or transfer to another checkpoint.

For benchmark design and controls, see the
[experiment guide](experiment-guide.md) and the
[experiment report specification](../specifications/experiment-report-v1.md).
