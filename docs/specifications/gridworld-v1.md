# Micro-World Gridworld v1

**Status:** Implemented baseline  
**Purpose:** Define the deterministic visual environment used before transformer integration.

## World contract

- World size: 16×16 cells.
- Time: one action advances exactly one fixed simulation tick.
- Day cycle: 100 ticks—day 0–59, dusk 60–69, night 70–89, dawn 90–99.
- Coordinates: `(0, 0)` is the north-west cell.
- Replay identity: seed, map name, configuration, ordered agent actions.
- Maps: one handcrafted `physics-lab` and seeded `procedural` maps.

Browser frame rate, animation, network latency, and wall-clock time never affect
the simulation.

## Agent observation

Each agent receives a 9×9 egocentric field of view. This is not an
agent-centred square.

```text
              forward
                 ^
    +-------------------------+
    | . . . . . . . . . | row 0
    | . . . . . . . . . |
    | . . . . . . . . . |
    | . . . . . . . . . |
    | . . . . . . . . . |
    | . . . . . . . . . |
    | . . . . . . . . . |
    | . . . . . . . . . |
    | . . . . A . . . . | row 8
    +-------------------------+
              local (4, 8)
```

The view rotates with the agent so forward is always upward. Walls and closed
doors are visible but occlude cells behind them. Cells outside the map, behind
occluders, outside natural night range, or otherwise imperceptible are encoded
as unknown.

Natural visibility is unrestricted within the field of view during day, five
cells during dusk and dawn, and three cells at night. Active lamps reveal cells
within three cells when line of sight is clear.

## Environmental action space

Environmental actions remain separate from the 128-token language:

```text
forward
turn-left
turn-right
interact
take
drop
wait
```

An action targets one agent and advances one tick. The language vocabulary is
unchanged by this release.

## Tile rules

| Tile | Rule |
|---|---|
| Floor | No automatic effect. |
| Wall | Blocks movement and vision. |
| Door | Closed doors block movement and vision; interaction toggles an unlocked door, while a locked door requires a carried key. |
| Water | Static shaded terrain in the baseline; it does not expand as the agent acts. |
| Ice | Preserves the entering movement direction and slides once on later ticks. |
| Directional floor | Pushes its occupant one cell in its marked direction per tick. |
| Clockwise spinner | Rotates an agent right once per tick. |
| Anticlockwise spinner | Rotates an agent left once per tick. |
| Pressure plate | Activates when occupied and opens unlocked doors. |
| Lamp | Provides a bounded visible area during reduced light. |

An optional flood experiment may enable one-cell-per-tick water propagation.
When enabled, closed doors block it and open doors allow it to reach the floor
beyond without replacing the door tile. It is disabled by default so ordinary
movement never appears to create terrain.

## Entities

The baseline contains two agents plus a key, cube, orb, and crate. Agents and
large objects block movement. Cubes, orbs, and crates may be pushed one cell if
the destination is free. Each agent initially carries at most one object.

## Tick order

```text
selected agent action
→ automatic ice/conveyor/spinner effect
→ pressure-plate activation
→ optional one-cell flood expansion (disabled by default)
→ clock and day-phase transition
→ perception and event output
```

Each transition emits structured events. Symbolic sound cues are derived from
those events; they do not influence world state.

## Visual observations

The canonical model observation is an exact 128×128 row-major RGB frame rendered
in Python without a game engine. The browser redraws the same structured
observation using Canvas, but browser pixels are not canonical training data.

The browser's primary view places the agent inside the 16×16 world and reveals
only cells present in its current 9×9 observation. Unknown and occluded cells
remain dark, so turning and cell-by-cell movement visibly move the field of view
through world coordinates. A separate smaller panel displays the exact rotated
9×9 model input.

The complete 16×16 state is available only through an explicit `Reveal debug
world` control. That full map must never be supplied to an agent unless an
experiment explicitly defines full observability.

## Version-one boundaries

Gridworld v1 does not change the existing language generator, train a model,
synthesize audio waveforms, implement continuous coordinates, or use browser
physics. Those capabilities remain separate extensions over this stable
environment contract.
