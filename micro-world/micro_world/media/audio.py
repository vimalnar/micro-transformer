"""Deterministic symbolic sound cues derived from simulation events."""

from __future__ import annotations

from dataclasses import dataclass

from micro_world.protocol import Event


@dataclass(frozen=True)
class AudioCue:
    tick: int
    sound: str
    position: tuple[int, int] | None
    intensity: int = 1

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "sound": self.sound,
            "position": list(self.position) if self.position else None,
            "intensity": self.intensity,
        }


SOUNDS = {
    "moved": "step",
    "blocked": "bump",
    "door-opened": "door",
    "door-closed": "door",
    "door-unlocked": "unlock",
    "taken": "pickup",
    "dropped": "drop",
    "water-spread": "water",
    "rotated": "click",
}


def cues_for_events(events: list[Event]) -> list[AudioCue]:
    """Translate events into exact sound categories; waveform synthesis comes later."""

    return [
        AudioCue(event.tick, SOUNDS[event.kind], event.position)
        for event in events
        if event.kind in SOUNDS
    ]

