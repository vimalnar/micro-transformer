# Micro-Transformer Language: Reference

A fixed-size formal English-like language for training and evaluating tiny neural models

| | |
|---|---|
| Author | Vimal Naran |
| Version | 1.0 (September 2026) |
| Contact | https://vimalnaran.substack.com |

## Purpose:

Micro-Transformer is a small English-like formal language for a simulated persistent world. Every valid statement has one defined effect, and every question has an exact answer from the interpreter. The language is intentionally small enough for a neural network model trained from scratch, while still supporting new events, delayed questions, conditional behavior, and memory experiments.

Even though Micro-Transformer uses English-like words for readability, unlike ordinary English it has fixed grammar and exact meanings, so a program can execute and validate every episode.

## Design Principles:

- Readable surface form. Sentences are easy for a person to inspect.
- Fixed meanings. Each token has one interpreter defined role.
- Compositionality. Familiar words combine into previously unseen episodes.
- Exact scoring. The interpreter executes statements and checks answers.
- Persistent state. Facts remain true until an action changes them.

## Experiment Extensions:

Keep the vocabulary and interpreter stable. Increase difficulty through episode construction.

- Hold out selected agent object location combinations during training.
- Ask questions after longer distractor sequences.
- Use new initial world layouts at inference time.
- Introduce a temporary local convention in each episode, then test rapid adaptation.
- Compare a context only baseline with persistent state, fast weight, and small adapter variants.

## Validation Checklist:

| Check | What the harness verifies |
|---|---|
| Syntax | Every sequence follows a valid grammar pattern. |
| Execution | Every action has legal preconditions and updates the world correctly. |
| Question answering | The response exactly matches the current world state. |
| Generalization | Test episodes contain held out combinations, states, and longer delays. |
| Adaptation | A model incorporates new episode facts or rules without overwriting prior rules. |

## Vocabulary:

The initial language uses 128 fixed, unique tokens. The list below is complete: the category totals and the distinct token total are both 128. Difficulty grows through longer, richer episodes rather than new words. Atomic punctuation is part of the vocabulary: . ends a statement, ? ends a question, and | ends a self contained episode and resets the harness. Use one canonical word order with no synonyms, verb conjugations, or category dependent capitalization. Padding and batch markers belong only to the training pipeline, not the visible language.

| Category | Count | Examples | Role |
|---|---:|---|---|
| Grammar and control | 16 | . &nbsp; ? &nbsp; \| &nbsp; then, if, else, not, and, to, from, before, after, because, while, only, again | Sentence structure, boundaries, sequence, and conditions |
| Actions | 16 | move, take, drop, give, open, close, lock, unlock, paint, hide, find, build, put, remove, swap, follow | Changes to world state |
| Questions and answers | 12 | where, who, what, which, is, count, colour, yes, no, unknown, does, can | Queries and exact responses |
| Relations and mental state | 12 | in, at, has, with, behind, beside, inside, contains, sees, knows, believes, tell | Facts, spatial relations, observation, belief, and communication |
| Attributes and states | 16 | red, blue, green, yellow, small, heavy, light, closed, locked, hidden, new, old, empty, full, visible, broken | Object description and state |
| Agent identifiers | 8 | ava, ben, cy, da, eri, fin, gia, hal | Actors |
| Object kinds | 16 | cube, key, orb, box, ball, book, gem, tool, gate, door, coin, flag, map, ring, lamp, chest | Types of objects |
| Object identifiers | 16 | one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen | Distinct objects, for example cube two |
| Locations | 12 | garden, hall, tower, cave, river, lab, store, home, bridge, room, forest, dock | Places |
| Quantities and modifiers | 4 | all, some, none, same | Broader statements and comparisons |

## Core Grammar:

The grammar is deliberately regular. The interpreter rejects malformed input rather than guessing intent.

| Form | Pattern | Example |
|---|---|---|
| Action | agent action object [to or from location] . | ava move cube two to garden . |
| Transfer | agent give object to agent . | ava give key one to ben . |
| State change | agent action object [attribute or location] . | eri paint orb three red . |
| Conditional | if condition then action . | if door one locked then ava unlock door one . |
| Question | question target ? | where cube two ? |
| Belief question | where agent believes object ? | where ben believes cube two ? |
| Episode boundary | final answer or final statement \| | garden \| |

## Interpreter Semantics:

The world harness is the source of meaning and validation. It maintains a state containing object locations, ownership, visible status, attributes, door and gate states, and agent relationships.

- move changes an object or agent location.
- take gives the acting agent possession of an object.
- drop places a possessed object at the agent location or named location.
- give transfers possession between agents.
- open, close, lock, and unlock change a gate, door, or box state.
- paint changes an object colour; hide changes visibility; find reverses hidden status when the preconditions are met.
- sees records that an agent observed an event. tell gives one agent a stated fact from another agent. knows and believes are stored separately for each agent.
- Questions inspect the current world state and return a canonical answer token sequence. The | token ends the episode and resets the harness for the next one.

## Agent Knowledge and Belief:

The harness can maintain a private belief state for every agent as well as the real world state. An agent updates its belief only when it observes an event or receives a statement. This allows a model to answer about what an agent believes, even when that belief is outdated or false.

| Token | Interpreter effect |
|---|---|
| sees | The observing agent receives the witnessed event in its private belief state. |
| tell | The receiving agent stores the stated fact as a belief. It may later be made unreliable for misinformation experiments. |
| knows | Marks a fact that is currently supported by an observation or validated statement. |
| believes | Queries or represents the agent private view, which may differ from the real world. |

### False belief scenario:

```text
ava move cube two to garden .
ben sees ava move cube two to garden .
ava move cube two to cave .
where ben believes cube two ?
garden |
```

The real location is cave. Ben did not observe the later move, so his stored belief remains garden. This is a compact theory of mind test.

### Communication scenario:

```text
ava move key one to tower .
ava tell ben key one at tower .
where ben believes key one ?
tower |
```

Ben learns the fact through communication rather than direct observation.

## Examples by Scenario:

### Movement and location:

```text
ava move cube two to garden .
where cube two ?
garden |
```

The interpreter records cube two at garden. The question must return garden.

### Ownership and transfer:

```text
ben take key one .
ben give key one to cy .
who has key one ?
cy |
```

The model must update the owner after both actions; the answer is not determined by the first event alone.

### State and property:

```text
eri paint orb three red .
what colour orb three ?
red |
fin lock door one .
is door one locked ?
yes |
```

### Conditional behavior:

```text
if door one locked then ava unlock door one .
is door one locked ?
no |
```

The conditional performs an action only when its condition is true. This creates a simple inference and planning task.

### Delayed question:

```text
ava move cube twelve to cave .
ben take key seven .
cy paint ball one blue .
da open box three .
where cube twelve ?
cave |
```

The relevant fact appears before unrelated events. This is a basic memory test.

### Multi step episode:

```text
ava take key one .
if ava has key one then ava open gate two .
ava move cube two to tower .
ben hide gem four in box two .
who has key one ?
ava
where cube two ?
tower
is gate two open ?
yes |
```

## Extensions:

Micro-Transformer v1.0 is a fixed 128-token language and benchmark. Future versions may add new vocabulary, grammar, world rules, and evaluation tasks where these support a specific experiment. Each extension should be versioned and documented, while the v1.0 vocabulary, interpreter, and test set remain unchanged for comparison.

Future extensions may introduce controlled new concepts and tasks, allowing models to be tested on increasingly complex reasoning, adaptation, and learning without changing the original benchmark.
