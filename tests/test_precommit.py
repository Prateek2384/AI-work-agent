from agent.loop import Agent, Criterion
from agent.trace import Trace
from llm.client import LLMReply, ToolCall
from tools.registry import Registry


def call(name, **args):
    return LLMReply(None, [ToolCall(f"id-{name}", name, args)])


class Scripted:
    def __init__(self, replies):
        self.r = list(replies)

    def chat(self, messages, tools=None, **kw):
        return self.r.pop(0)


class FakeVerifier:
    def __init__(self, reviews, final=None):
        self.reviews, self.final = list(reviews), final or [Criterion("ok", True, "")]
        self.reviewed = 0

    def review_commit(self, goal, criteria, transcript, preview):
        self.reviewed += 1
        return self.reviews.pop(0) if self.reviews else (True, "")

    def check(self, goal, memory, transcript, criteria=None, claim=""):
        return self.final


def make(tmp_path, replies, reviews, result="saved", **kw):
    reg, saved = Registry(), []

    @reg.tool("submit", "write", {"properties": {"text": {"type": "string"}}}, risk="write")
    def submit(text="x"):
        saved.append(text)
        return result

    v = FakeVerifier(reviews, kw.pop("final", None))
    agent = Agent(Scripted(replies), reg, Trace(tmp_path), verifier=v,
                  commit_preview=lambda name, args: "submitting a form with item=A" if name == "submit" else None, **kw)
    return agent, saved, v


def test_save_is_refused_until_the_review_passes(tmp_path):
    agent, saved, _ = make(tmp_path, [call("submit"), call("submit"), call("finish", summary="done")],
                           [(False, "that is not the oldest item"), (True, "")])
    res = agent.run("t")
    assert saved == ["x"] and res.status == "done"  # first attempt blocked, second saved exactly once


def test_nothing_is_saved_when_the_review_keeps_failing_and_the_user_declines(tmp_path):
    agent, saved, _ = make(tmp_path, [call("submit")] * 3 + [call("finish", summary="gave up")],
                           [(False, "wrong")] * 5, approve=lambda n, a: False)
    agent.run("t")
    assert saved == []


def test_user_can_override_after_repeated_refusals(tmp_path):
    asked = []
    agent, saved, _ = make(tmp_path, [call("submit")] * 3 + [call("finish", summary="ok")],
                           [(False, "wrong")] * 5, approve=lambda n, a: asked.append((n, a)) or True)
    agent.run("t")
    assert saved == ["x"] and asked[0][0] == "save_form" and "wrong" in asked[0][1]["concern"]


def test_multiple_saves_are_disclosed_in_the_final_answer(tmp_path):
    agent, saved, _ = make(tmp_path, [call("submit"), call("submit"), call("finish", summary="all done")], [])
    res = agent.run("t")
    assert len(saved) == 2 and "Records saved by this run (2)" in res.summary


def test_failed_http_save_is_not_counted_as_a_record(tmp_path):
    agent, _, _ = make(tmp_path, [call("submit"), call("finish", summary="d")], [], result="page (HTTP 422)")
    assert agent.run("t").summary == "d"  # no disclosure noise: nothing was actually saved


def test_unverified_run_says_so(tmp_path):
    agent, _, _ = make(tmp_path, [call("finish", summary="done")], [],
                       final=[Criterion("UNVERIFIED", True, "auditor gave nothing")])
    res = agent.run("t")
    assert res.status == "done" and "Not independently verified" in res.summary


def test_no_review_without_a_verifier_or_preview_hook(tmp_path):
    reg, saved = Registry(), []

    @reg.tool("submit", "w", {"properties": {}}, risk="write")
    def submit():
        saved.append(1)
        return "ok"

    Agent(Scripted([call("submit"), call("finish", summary="d")]), reg, Trace(tmp_path)).run("t")
    assert saved == [1]
