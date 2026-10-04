from agent.loop import Agent
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


def make(tmp_path, replies):
    reg, entered = Registry(), []

    @reg.tool("read_all_pages", "listing")
    def read_all_pages():
        return LISTING

    @reg.tool("enter", "write", {"properties": {"text": {"type": "string"}}, "required": ["text"]}, risk="write")
    def enter(text):
        entered.append(text)
        return "ok"

    return Agent(Scripted(replies), reg, Trace(tmp_path)), entered


def test_trimming_never_cuts_document_text_but_trims_page_dumps():
    from agent.loop import _trim_old_observations
    doc = "x" * 800 + " Total Amount Payable: INR 91,000.00"
    page = "URL: http://x\n--- PAGE TEXT ---\n" + "y" * 800 + "\n--- INTERACTIVE ELEMENTS ---\n[e1] a"
    msgs = [{"role": "tool", "tool_call_id": "1", "content": doc}, {"role": "tool", "tool_call_id": "2", "content": page}] + \
           [{"role": "tool", "tool_call_id": str(i), "content": "ok"} for i in range(3, 6)]
    listing = "--- page 1 (http://x) ---\n" + "z" * 800
    out = _trim_old_observations([{"role": "tool", "tool_call_id": "0", "content": listing}] + msgs)
    assert out[0]["content"] == listing and out[1]["content"] == doc  # lists and documents are never trimmed
    assert "older output trimmed" in out[2]["content"]  # single-page browser dumps are


def test_finish_refused_while_form_input_is_unsubmitted(tmp_path):
    state = {"pending": "you typed into 2 form field(s) but never submitted"}
    agent, _ = make(tmp_path, [call("finish", summary="done"), call("finish", summary="done")])
    notes = iter([state["pending"], None])
    agent.pending_check = lambda: next(notes)  # pending at the first finish, submitted by the second
    res = agent.run("Enter invoice INV-1")
    assert res.status == "done" and res.steps == 2


def test_finish_refused_while_form_input_is_unsubmitted(tmp_path):
    state = {"pending": "you typed into 2 form field(s) but never submitted"}
    agent, _ = make(tmp_path, [call("finish", summary="done"), call("finish", summary="done")])
    notes = iter([state["pending"], None])
    agent.pending_check = lambda: next(notes)  # pending at the first finish, submitted by the second
    res = agent.run("Enter invoice INV-1")
    assert res.status == "done" and res.steps == 2
