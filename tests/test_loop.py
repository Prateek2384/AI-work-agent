from agent.loop import Agent, Criterion
from agent.trace import Trace
from llm.client import LLMReply, ToolCall
from tools.registry import IRREVERSIBLE, Registry, ToolError, TRANSIENT


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools=None, **kw):
        return self.replies.pop(0)


def call(name, **args):
    return LLMReply(None, [ToolCall(f"id-{name}", name, args)])


def make(tmp_path, replies, **kw):
    reg = Registry()
    state = {"n": 0, "paid": False}

    @reg.tool("flaky", "fails once")
    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise ToolError("503", TRANSIENT)
        return "ok"

    @reg.tool("pay", "irreversible", risk=IRREVERSIBLE)
    def pay():
        state["paid"] = True
        return "paid"

    agent = Agent(ScriptedLLM(replies), reg, Trace(tmp_path), **kw)
    return agent, state


def test_retry_then_finish(tmp_path):
    agent, state = make(tmp_path, [call("flaky"), call("flaky"), call("finish", summary="done")])
    res = agent.run("t")
    assert res.status == "done" and state["n"] == 2


def test_irreversible_denied_by_default(tmp_path):
    agent, state = make(tmp_path, [call("pay"), call("finish", summary="blocked")])
    agent.run("t")
    assert state["paid"] is False


def test_irreversible_runs_when_approved(tmp_path):
    agent, state = make(tmp_path, [call("pay"), call("finish", summary="ok")], approve=lambda n, a: True)
    agent.run("t")
    assert state["paid"] is True


def test_failed_verification_blocks_done(tmp_path):
    class V:
        calls = 0

        def check(self, goal, memory, transcript, criteria=None, claim=""):
            V.calls += 1
            return [Criterion("saved", V.calls > 1, "record missing")]

    agent, _ = make(tmp_path, [call("finish", summary="x"), call("finish", summary="y")], verifier=V())
    res = agent.run("t")
    assert res.status == "done" and V.calls == 2


def test_budget_exhausted(tmp_path):
    agent, _ = make(tmp_path, [call("flaky")] * 5, max_steps=3)
    assert agent.run("t").status == "budget_exhausted"


def test_memory_roundtrip(tmp_path):
    agent, _ = make(tmp_path, [call("remember", key="amount", value="120.50"), call("finish", summary="s")])
    agent.run("t")
    assert agent.memory.get("amount") == "120.50"


def test_llm_failure_aborts_cleanly(tmp_path):
    from llm.client import LLMQuotaError

    class Dead:
        def chat(self, messages, tools=None, **kw):
            raise LLMQuotaError("daily quota exhausted")

    res = Agent(Dead(), Registry(), Trace(tmp_path)).run("t")
    assert res.status == "aborted" and "unavailable" in res.summary


def test_guard_blocks_invented_numbers_but_allows_observed_ones(tmp_path):
    from agent.loop import ungrounded_numbers
    msgs = [{"role": "system", "content": "notes"}, {"role": "user", "content": "Task: enter the bill"},
            {"role": "tool", "tool_call_id": "1", "content": "Total Amount Payable: INR 48,250.00 INV-2047"}]
    assert ungrounded_numbers({"text": "48250"}, msgs) == []          # observed (commas/.00 normalised)
    assert ungrounded_numbers({"text": "48250.00"}, msgs) == []
    assert ungrounded_numbers({"text": "INV-2047"}, msgs) == []
    assert ungrounded_numbers({"text": "15000"}, msgs) == ["15000"]    # invented
    assert ungrounded_numbers({"text": "Acme", "element_id": "e12345"}, msgs) == []  # ids and words ignored
    assert ungrounded_numbers({"text": "12"}, msgs) == []              # short numbers are not amounts


def test_agent_refuses_write_with_invented_number(tmp_path):
    reg = Registry()
    seen = []

    @reg.tool("enter", "write", {"properties": {"text": {"type": "string"}}, "required": ["text"]}, risk="write")
    def enter(text):
        seen.append(text)
        return "ok"

    replies = [call("enter", text="15000"), call("finish", summary="gave up")]
    Agent(ScriptedLLM(replies), reg, Trace(tmp_path)).run("t")
    assert seen == []
