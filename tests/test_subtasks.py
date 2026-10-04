from agent.loop import Agent
from agent.planner import make_plan
from agent.trace import Trace
from llm.client import LLMReply, ToolCall
from tools.registry import Registry


def call(name, **args):
    return LLMReply(None, [ToolCall(f"id-{name}", name, args)])


class Scripted:
    def __init__(self, replies):
        self.r, self.seen = list(replies), []

    def chat(self, messages, tools=None, **kw):
        self.seen.append(messages)
        return self.r.pop(0)


def plan_reply(subtasks):
    import json
    return LLMReply(json.dumps({"kind": "action", "success_criteria": ["c"], "approach": ["a"], "subtasks": subtasks}))


def test_planner_returns_subtasks_only_when_splitting_is_allowed():
    llm = Scripted([plan_reply(["do A in full", "do B in full"]), plan_reply(["do A in full", "do B in full"])])
    assert make_plan(llm, "A and B")["subtasks"] == ["do A in full", "do B in full"]
    assert make_plan(llm, "A", allow_split=False)["subtasks"] == []


def test_multi_part_request_runs_each_part_in_a_fresh_conversation(tmp_path):
    llm = Scripted([plan_reply(["part one", "part two"]),                  # overall plan
                    plan_reply([]), call("remember", key="k", value="v"), call("finish", summary="one done"),
                    plan_reply([]), call("finish", summary="two done")])   # sub-plan, work, per part
    agent = Agent(llm, Registry(), Trace(tmp_path), use_planner=True)
    res = agent.run("part one and part two")
    assert res.status == "done" and "1. one done" in res.summary and "2. two done" in res.summary
    # the second part starts with a clean conversation: it does not contain the first part's tool calls
    second = [m for m in llm.seen if m and m[0]["role"] == "system" and "Task: part two" in str(m[-1])]
    assert second and not any("remember" in str(m) for m in second[0])
    assert agent.memory.get("k") == "v"                                  # ...but memory is shared


def test_a_failed_part_stops_the_run_and_says_what_was_skipped(tmp_path):
    llm = Scripted([plan_reply(["part one", "part two"]), plan_reply([])] + [call("remember", key="k", value="v")] * 3)
    res = Agent(llm, Registry(), Trace(tmp_path), use_planner=True, max_steps=3).run("x")
    assert res.status == "budget_exhausted" and "later part(s) were not attempted" in res.summary


def test_single_item_request_is_not_split(tmp_path):
    llm = Scripted([plan_reply(["just the task"]), call("finish", summary="done")])
    res = Agent(llm, Registry(), Trace(tmp_path), use_planner=True).run("just the task")
    assert res.status == "done" and res.summary == "done"


def test_later_parts_receive_the_results_of_earlier_parts(tmp_path):
    llm = Scripted([plan_reply(["find X", "use X"]),
                    plan_reply([]), call("finish", summary="X is INV-7, 100.00"),
                    plan_reply([]), call("finish", summary="used it")])
    Agent(llm, Registry(), Trace(tmp_path), use_planner=True).run("find X and use X")
    second = next(m for m in llm.seen if m and m[0]["content"].startswith("You are an autonomous AI worker") and "Task: use X" in str(m[1]))
    assert "X is INV-7, 100.00" in second[0]["content"] and "do not ask for them" in second[0]["content"]


def test_questions_can_be_switched_off(tmp_path):
    asked = []
    llm = Scripted([call("finish", summary="d"), call("finish", summary="d")])
    off = Agent(llm, Registry(), Trace(tmp_path), allow_questions=False, ask_user=lambda q: asked.append(q) or "x")
    assert "ask_user" not in off.registry.tools
    assert "CANNOT ask the user" in (off.run("t") and llm.seen[0][0]["content"])
    on = Agent(Scripted([call("finish", summary="d")]), Registry(), Trace(tmp_path))
    assert "ask_user" in on.registry.tools
