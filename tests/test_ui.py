"""Chat UI protocol test with a scripted LLM and a fake browser (no Chrome, no network)."""
import ui.server as server
from fastapi.testclient import TestClient
from llm.client import LLMReply, ToolCall
from tools.registry import Registry


class FakeBrowser:
    def close(self):
        pass


PLAN = LLMReply('{"kind": "action", "success_criteria": ["it is done"], "approach": ["do it"]}')
VERDICT = LLMReply('{"criteria": [{"name": "it is done", "passed": true, "evidence": "seen"}]}')


def call(name, **args):
    return LLMReply(None, [ToolCall(f"id-{name}", name, args)])


def patch(monkeypatch, tmp_path, replies):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "build_tools", lambda *a, **k: (Registry(), FakeBrowser(), ""))

    class Scripted:
        def __init__(self):
            self.r = list(replies)

        def chat(self, messages, tools=None, **kw):
            return self.r.pop(0)

    monkeypatch.setattr(server, "LLMClient", Scripted)


def collect(ws, stop_types):
    events = []
    while True:
        ev = ws.receive_json()
        events.append(ev)
        if ev["type"] in stop_types:
            return events


def test_task_streams_steps_and_result(monkeypatch, tmp_path):
    patch(monkeypatch, tmp_path, [PLAN, call("remember", key="k", value="v"), call("finish", summary="all good"), VERDICT])
    with TestClient(server.create_app(start_env=False)) as c, c.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "status", "running": False}
        ws.send_json({"type": "task", "text": "do it"})
        events = collect(ws, {"done"})
    assert any(e["type"] == "trace" and e["event"] == "tool" and e["name"] == "remember" for e in events)
    done = events[-1]
    assert done["status"] == "done" and done["summary"] == "all good"


def test_ask_user_roundtrip(monkeypatch, tmp_path):
    patch(monkeypatch, tmp_path, [PLAN, call("ask_user", question="Which Acme?"), call("finish", summary="ok"), VERDICT])
    with TestClient(server.create_app(start_env=False)) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "task", "text": "t", "allow_questions": True})
        events = collect(ws, {"ask"})
        assert events[-1]["question"] == "Which Acme?"
        ws.send_json({"type": "reply", "text": "Acme Supplies"})
        done = collect(ws, {"done"})[-1]
    assert done["status"] == "done"


def test_stop_cancels_run(monkeypatch, tmp_path):
    patch(monkeypatch, tmp_path, [PLAN, call("ask_user", question="?")])
    with TestClient(server.create_app(start_env=False)) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "task", "text": "t", "allow_questions": True})
        collect(ws, {"ask"})
        ws.send_json({"type": "stop"})
        done = collect(ws, {"done"})[-1]
    assert done["status"] == "aborted"
