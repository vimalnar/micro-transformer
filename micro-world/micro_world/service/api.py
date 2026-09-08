"""FastAPI application for the optional browser inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from micro_world.media import cues_for_events, render_png
from micro_world.protocol import WorldConfig

from .sessions import SessionManager


def create_app():
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, Response
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError(
            "Browser dependencies are not installed. Run: "
            "python3 -m pip install -e '.[web]'"
        ) from exc

    project_root = Path(__file__).resolve().parents[3]
    inspector = project_root / "micro-world" / "app"
    manager = SessionManager()
    app = FastAPI(title="Micro-World Inspector API", version="0.1.0")
    app.mount("/static", StaticFiles(directory=inspector), name="static")

    def session_or_404(session_id: str):
        try:
            return manager.get(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def session_payload(session, agent_id: str = "ava") -> dict:
        observation = session.world.observe(agent_id)
        return {
            "session_id": session.id,
            "state": session.world.state.to_dict(),
            "observation": observation.to_dict(),
            "events": [event.to_dict() for event in session.world.events[-50:]],
            "replay": session.world.replay_record(),
        }

    @app.get("/")
    async def index():
        return FileResponse(inspector / "index.html")

    @app.post("/sessions")
    async def create_session(payload: Optional[dict] = None):
        body = payload or {}
        try:
            session = manager.create(
                int(body.get("seed", 42)),
                str(body.get("map_name", "physics-lab")),
                WorldConfig(),
            )
            return session_payload(session, str(body.get("agent_id", "ava")))
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/reset")
    async def reset_session(session_id: str, payload: Optional[dict] = None):
        body = payload or {}
        session_or_404(session_id)
        try:
            session = manager.reset(
                session_id,
                int(body.get("seed", 42)),
                str(body.get("map_name", "physics-lab")),
                WorldConfig(),
            )
            return session_payload(session, str(body.get("agent_id", "ava")))
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/step")
    async def step_session(session_id: str, payload: dict):
        session_or_404(session_id)
        try:
            agent_id = str(payload.get("agent_id", "ava"))
            result = manager.step(session_id, str(payload["action"]), agent_id)
            return {
                "session_id": session_id,
                **result.to_dict(),
                "audio": [cue.to_dict() for cue in cues_for_events(result.events)],
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/state")
    async def get_state(session_id: str, agent_id: str = "ava"):
        return session_payload(session_or_404(session_id), agent_id)

    @app.get("/sessions/{session_id}/observation/{agent_id}")
    async def get_observation(session_id: str, agent_id: str):
        session = session_or_404(session_id)
        try:
            return session.world.observe(agent_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/observation/{agent_id}/frame.png")
    async def get_observation_png(session_id: str, agent_id: str):
        session = session_or_404(session_id)
        try:
            return Response(
                render_png(session.world.observe(agent_id)), media_type="image/png"
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/events")
    async def get_events(session_id: str):
        session = session_or_404(session_id)
        return [event.to_dict() for event in session.world.events]

    @app.websocket("/sessions/{session_id}/stream")
    async def stream(websocket, session_id: str):
        try:
            session = manager.get(session_id)
        except KeyError:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        try:
            await websocket.send_json(session_payload(session))
            while True:
                message = await websocket.receive_json()
                command = message.get("command", "state")
                agent_id = str(message.get("agent_id", "ava"))
                if command == "step":
                    result = manager.step(session_id, str(message["action"]), agent_id)
                    await websocket.send_json(
                        {
                            "session_id": session_id,
                            **result.to_dict(),
                            "audio": [
                                cue.to_dict() for cue in cues_for_events(result.events)
                            ],
                        }
                    )
                else:
                    await websocket.send_json(session_payload(session, agent_id))
        except WebSocketDisconnect:
            return

    return app


app = None
try:  # Lets uvicorn import the module while keeping core installs dependency-free.
    app = create_app()
except RuntimeError:
    pass
