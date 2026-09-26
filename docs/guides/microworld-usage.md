# Using Micro-World

Micro-World is a deterministic, top-down grid simulation with typed state,
discrete actions, partial agent observations, events, replay records, a visual
renderer, sound-cue descriptions, an HTTP/WebSocket service and a browser
inspector. It is implemented separately from the formal-language generator and
the transformer.

## Start the inspector

Install the core package's development extra, then run from the repository root:

```bash
.venv/bin/python scripts/run_inspector.py
```

Open <http://127.0.0.1:8000>. The included inspector lets a person view the map,
select an agent, step or reset the simulation, and inspect observations/events.
The full map is a debugging surface. A model or agent receives an
agent-specific `Observation`, not the complete state automatically.

## World and agent data

The typed interface separates:

- `WorldState`: authoritative simulation state;
- `Action`: a discrete action request;
- `Observation`: an agent-local view with visible cells, entities, direction,
  tick and day phase;
- `Event`: a record of state changes or interactions;
- `StepResult`: the resulting state, observation and events.

Actions are `forward`, `turn-left`, `turn-right`, `interact`, `take`, `drop`, and
`wait`. They are environment action names, not tokens in the 128-token language.
The exact effect depends on current orientation, tile and entity rules specified
in [Gridworld v1](../specifications/gridworld-v1.md).

The simulation can be reset from a seed and map name. A replay record contains
the information needed to identify the scenario and its action sequence; it is
useful for deterministic regression and debugging. It does not by itself show
that a behavioural outcome is a valid causal experiment.

## HTTP interface

The optional FastAPI app exposes these implemented routes:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serve the inspector page. |
| `POST` | `/sessions` | Create an in-memory session; accepts optional seed, map name and agent ID. |
| `POST` | `/sessions/{id}/reset` | Reset that session with a seed/map. |
| `POST` | `/sessions/{id}/step` | Apply one named action. |
| `GET` | `/sessions/{id}/state` | Return serialized state, default agent observation, recent events and replay record. |
| `GET` | `/sessions/{id}/observation/{agent_id}` | Return a specific agent's observation. |
| `GET` | `/sessions/{id}/observation/{agent_id}/frame.png` | Render the agent observation as PNG. |
| `GET` | `/sessions/{id}/events` | Return the session event history. |
| `WS` | `/sessions/{id}/stream` | Send current session data and accept step/state messages. |

Sessions are held in process memory. This service is a local interactive/research
interface, not a production hosted service: no persistent session database,
authentication, authorization layer, multi-process coordination or deployment
hardening is supplied by this code.

## Create a session and step it

Start the inspector service in one terminal, then create a session from another:

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -H 'Content-Type: application/json' \
  -d '{"seed":42,"map_name":"physics-lab","agent_id":"ava"}'
```

The response includes a generated `session_id`, serialized state, the selected
agent's observation, recent events, and a replay record. Use that returned ID in
the next request:

```bash
curl -X POST http://127.0.0.1:8000/sessions/SESSION_ID/step \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"ava","action":"forward"}'
```

Replace `SESSION_ID` with the value returned by the first request. A step response
contains the resulting state and observation, events from that step, and symbolic
audio cues derived from those events. The state endpoint is for inspection; an
agent-facing experiment should consume only the declared observation.

## Write an agent or model adapter

An adapter should read the allowed `Observation`, select an action, and submit it
through the `World` or service interface. It must not directly mutate world state.
Record the observation, action, result, seed/map, and replay information if the
run needs to be reproduced.

The bundled language model is not trained as a Micro-World action policy. The
repository's `micro_world.agent_adapter` and `micro_world.benchmark` are bounded
research interfaces; the v1 benchmark deliberately reports parser/task failures,
controls and negative results. A feasibility script or reactive baseline does not
establish that the language model can control the world. See
[`micro-world-model-adapter.md`](micro-world-model-adapter.md) and
[`micro-world-reference-v1.md`](../results/micro-world-reference-v1.md).

## Visual and audio outputs

The Python renderer produces a low-resolution RGB representation or PNG from the
agent observation. The browser inspector draws a human-facing view with Canvas.
Simulation events can be converted to symbolic sound cues; the cue data is not a
trained audio model or a claim that an agent hears rendered audio. These media
outputs do not alter simulation state or reveal hidden state beyond their input
observation.
