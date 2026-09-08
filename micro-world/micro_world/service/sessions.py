"""In-memory interactive sessions over the authoritative World API."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from micro_world.protocol import Action, WorldConfig
from micro_world.simulation import World


@dataclass
class Session:
    id: str
    world: World
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(
        self,
        seed: int = 42,
        map_name: str = "physics-lab",
        configuration: WorldConfig | None = None,
    ) -> Session:
        session = Session(
            id=uuid.uuid4().hex,
            world=World.reset(seed, configuration, map_name),
        )
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc

    def reset(
        self,
        session_id: str,
        seed: int,
        map_name: str,
        configuration: WorldConfig | None = None,
    ) -> Session:
        session = self.get(session_id)
        with session.lock:
            session.world = World.reset(seed, configuration, map_name)
        return session

    def step(self, session_id: str, action: str | Action, agent_id: str = "ava"):
        session = self.get(session_id)
        with session.lock:
            return session.world.step(action, agent_id)

