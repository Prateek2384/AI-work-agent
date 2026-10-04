"""Chat UI: `python -m ui.server` then open http://127.0.0.1:8000

One WebSocket carries the whole conversation. Browser → server: task / reply / approval / stop / reset.
Server → browser: step events, ask/approve prompts, and the final result.
The agent runs in a worker thread (Playwright's sync API needs its own thread)."""
from __future__ import annotations

import asyncio
import os
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from agent.loop import Agent
from agent.trace import Trace
from agent.verifier import LLMVerifier
from env import erp
from env.run import ERP_URL, PORTAL_URL, start_all, stop_all
from llm.client import LLMClient
from tools.setup import build_tools

INDEX = Path(__file__).parent / "index.html"
REPLY_TIMEOUT = 600  # seconds to wait for the user to answer a question / approval


class Session:
    """Single local user: one run at a time."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.out: asyncio.Queue | None = None
        self.inbox: queue.Queue = queue.Queue()
        self.agent: Agent | None = None
        self.running = False

    def emit(self, ev: dict) -> None:
        if self.loop and self.out is not None:
            self.loop.call_soon_threadsafe(self.out.put_nowait, ev)

    def _wait(self, kind: str):
        try:
            while True:
                msg = self.inbox.get(timeout=REPLY_TIMEOUT)
                if msg.get("type") == kind or msg.get("type") == "stop":
                    return msg
        except queue.Empty:
            return {"type": "stop"}

    def approve(self, name: str, args: dict) -> bool:
        self.emit({"type": "approve", "name": name, "args": args})
        return bool(self._wait("approval").get("ok"))

    def ask(self, question: str) -> str:
        self.emit({"type": "ask", "question": question})
        msg = self._wait("reply")
        return msg.get("text") or "(the user did not answer; proceed with the safest assumption or stop)"

    def run_task(self, text: str, visible: bool, max_steps: int, allow_questions: bool = False) -> None:
        self.running = True
        self.emit({"type": "status", "running": True})
        browser = None
        try:
            trace = Trace(sink=lambda ev, d: self.emit({"type": "trace", "event": ev, **d}))
            registry, browser, notes = build_tools(trace.dir, PORTAL_URL, ERP_URL, headless=not visible)
            llm = LLMClient()
            self.agent = Agent(llm, registry, trace, max_steps=max_steps, context=notes, use_planner=True,
                               verifier=LLMVerifier(llm) if os.environ.get("VERIFY", "1") != "0" else None, approve=self.approve, ask_user=self.ask, allow_questions=allow_questions,
                               pending_check=getattr(browser, "unsubmitted", None),
                               commit_preview=getattr(browser, "commit_preview", None))
            before = [dict(b) for b in erp.BILLS]
            res = self.agent.run(text)
            changed = [b for b in erp.BILLS if b not in before]  # only what THIS task added/changed
            self.emit({"type": "done", "status": res.status, "summary": res.summary, "steps": res.steps,
                       "trace_dir": res.trace_dir, "bills": changed,
                       "criteria": [c.__dict__ for c in res.criteria]})
        except Exception as e:  # never leave the UI hanging
            self.emit({"type": "done", "status": "error", "summary": f"{type(e).__name__}: {e}",
                       "steps": 0, "trace_dir": "", "bills": [], "criteria": []})
        finally:
            if browser:
                browser.close()
            self.agent, self.running = None, False
            self.emit({"type": "status", "running": False})


def create_app(start_env: bool = True) -> FastAPI:
    session = Session()

    @asynccontextmanager
    async def lifespan(_app):
        servers = start_all() if start_env else []
        yield
        stop_all(servers)

    app = FastAPI(lifespan=lifespan)
    app.state.session = session

    @app.get("/")
    def index():
        return FileResponse(INDEX)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        session.loop, session.out = asyncio.get_running_loop(), asyncio.Queue()
        await sock.send_json({"type": "status", "running": session.running})

        async def pump():
            while True:
                await sock.send_json(await session.out.get())

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                msg = await sock.receive_json()
                kind = msg.get("type")
                if kind == "task" and not session.running and (msg.get("text") or "").strip():
                    threading.Thread(target=session.run_task, daemon=True,
                                     args=(msg["text"].strip(), bool(msg.get("visible")),
                                           int(msg.get("max_steps") or 60), bool(msg.get("allow_questions")))).start()
                elif kind == "stop":
                    if session.agent:
                        session.agent.cancel.set()
                    session.inbox.put({"type": "stop"})
                elif kind in ("reply", "approval"):
                    session.inbox.put(msg)
                elif kind == "reset" and not session.running:
                    erp.reset()
                    await sock.send_json({"type": "info", "text": "ERP data reset."})
        except WebSocketDisconnect:
            pass
        finally:
            pump_task.cancel()
            session.loop = session.out = None

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    print("Chat UI → http://127.0.0.1:8000   (vendor portal :8001, ERP :8002)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
