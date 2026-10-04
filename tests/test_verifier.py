from agent.verifier import LLMVerifier, digest, evidence_items, grounding


def convo(observation):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "collect_documents", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": observation},
    ]


class Judge:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, tools=None, **kw):
        from llm.client import LLMReply
        return LLMReply(self.reply)


def test_grounding_flags_invented_values():
    items = evidence_items(convo("INV-5521 Total Amount Payable: INR 91,000.00"))
    assert grounding("INV-5521 is INR 91,000.00", items).passed
    bad = grounding("INV-5521 is INR 81,900.00 and INV-9999", items)
    assert not bad.passed and "INV-9999" in bad.evidence and "81900.00" in bad.evidence


def test_judge_verdicts_are_returned_and_grounding_included():
    j = Judge('{"criteria": [{"name": "all pages read", "evidence": "only page 1", "verdict": "FAIL"}]}')
    res = LLMVerifier(j).check("t", None, convo("INV-1 INR 5.00"), ["all pages read"], "INV-1 is INR 5.00")
    names = {c.name: c.passed for c in res}
    assert names["answer_grounded_in_observations"] is True and names["all pages read"] is False


def test_unparseable_judge_degrades_visibly():
    res = LLMVerifier(Judge("sorry")).check("t", None, convo("x"), None, "nothing numeric")
    assert any(c.name == "UNVERIFIED" for c in res)


def test_digest_drops_element_lists_and_caps_size():
    items = evidence_items(convo("page text\n--- INTERACTIVE ELEMENTS ---\n[e1] button"))
    assert "[e1]" not in digest(items)
    assert len(digest([("read_page", "", "x" * 5000)] * 30)) <= 9000


def test_grounding_pairs_amounts_with_their_invoice():
    obs = ("## INV-5510 Globex Corp 2026-08-11\nINV-5510.pdf: Total Amount Payable: INR 78,200.50\n"
           "## INV-5521 Globex Corp 2026-09-05\nINV-5521.pdf: Total Amount Payable: INR 91,000.00")
    items = evidence_items(convo(obs))
    assert grounding("INV-5510: INR 78,200.50\nINV-5521: INR 91,000.00", items).passed
    swapped = grounding("INV-5510: INR 91,000.00\nINV-5521: INR 91,000.00", items)
    assert not swapped.passed and "INV-5510" in swapped.evidence


class Seq:
    """Auditor stand-in that returns a fixed sequence of replies."""
    def __init__(self, replies):
        self.r = list(replies)
        self.calls = 0

    def chat(self, messages, tools=None, **kw):
        from llm.client import LLMReply
        self.calls += 1
        return LLMReply(self.r.pop(0) if self.r else "garbage")


def test_json_is_found_inside_fences_prose_and_think_blocks():
    from agent.planner import extract_json
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    assert extract_json('<think>{"x": 0}</think>{"a": 3}') == {"a": 3}
    assert extract_json("no json here") is None


def test_auditor_retries_garbage_and_then_succeeds():
    good = '{"criteria": [{"name": "c", "evidence": "e", "verdict": "PASS"}]}'
    j = Seq(["not json", "still not", good])
    res = LLMVerifier(j).check("t", None, convo("x"), ["c"], "")
    assert j.calls == 3 and [c.name for c in res] == ["c"] and res[0].passed


def test_auditor_that_never_answers_is_reported_as_unverified_not_silently_passed():
    res = LLMVerifier(Seq([])).check("t", None, convo("x"), ["c"], "")
    assert res[-1].name == "UNVERIFIED" and "no usable verdict" in res[-1].evidence


def test_review_commit_verdicts():
    ok, why = LLMVerifier(Seq(['{"evidence": "dates compared", "verdict": "PASS", "fix": ""}'])).review_commit(
        "t", ["c"], convo("x"), "submitting a form with a=1")
    assert ok
    ok, why = LLMVerifier(Seq(['{"evidence": "INV-2 is older", "verdict": "FAIL", "fix": "use INV-2"}'])).review_commit(
        "t", ["c"], convo("x"), "submitting a form with a=1")
    assert not ok and "INV-2 is older" in why and "use INV-2" in why
    ok, why = LLMVerifier(Seq([])).review_commit("t", None, convo("x"), "p")
    assert ok and why.startswith("UNVERIFIED")  # cannot judge: allow, but say so


class Broken:
    def chat(self, messages, tools=None, **kw):
        from llm.client import LLMError
        raise LLMError("model timed out")


def test_a_dead_auditor_degrades_to_unverified_instead_of_crashing():
    res = LLMVerifier(Broken()).check("t", None, convo("x"), ["c"], "")
    assert res[-1].name == "UNVERIFIED"
    ok, why = LLMVerifier(Broken()).review_commit("t", None, convo("x"), "p")
    assert ok and why.startswith("UNVERIFIED")


def test_a_dead_planner_means_no_plan_not_a_crash():
    from agent.planner import make_plan
    assert make_plan(Broken(), "task") is None


def test_digest_keeps_source_documents_even_after_many_later_steps():
    from agent.verifier import digest
    doc = ("collect_documents", "{}", "## INV-2047 row\nINV-2047.pdf: Total Amount Payable: INR 48,250.00")
    later = [("click", "{}", "URL: http://x\n--- PAGE TEXT ---\n" + "p" * 650)] * 30     # 30 routine pages
    text = digest([doc] + later)
    assert "INR 48,250.00" in text and len(text) <= 9000


def test_digest_drops_trivial_confirmations_and_the_harness_own_refusals():
    from agent.verifier import digest, evidence_items
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
                {"id": str(i), "type": "function", "function": {"name": "type_text", "arguments": "{}"}} for i in range(3)]},
            {"role": "tool", "tool_call_id": "0", "content": "typed into e3"},
            {"role": "tool", "tool_call_id": "1", "content": "NOT saved - pre-save review found a problem: x"},
            {"role": "tool", "tool_call_id": "2", "content": "ERROR[precommit]: NOT saved - something"}]
    assert evidence_items(msgs) == [("type_text", "{}", "typed into e3")]
    assert digest(evidence_items(msgs)) == ""


def test_planner_accepts_subtasks_given_as_objects():
    import json
    from agent.planner import make_plan
    reply = json.dumps({"kind": "action", "success_criteria": ["c"], "approach": ["a"],
                        "subtasks": [{"task": "do A end to end"}, "do B end to end"]})
    assert make_plan(Seq([reply]), "A and B")["subtasks"] == ["do A end to end", "do B end to end"]
