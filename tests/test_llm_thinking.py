"""A model that only thought has not answered, and must not be reported as mute.

``num_predict`` caps everything a reasoning model emits, thinking included, so a
budget sized for an answer is one a thinker can burn through before writing a word.
The reply then comes back with prose in ``thinking``, empty content, and
done_reason "length", and the caller logs "returned no text (empty response)" —
which reads like a broken model rather than a budget that cut it off.

Reproduced directly against a local qwen3.6, and pinned live at the bottom of this
file: at num_predict 16 with thinking on the content is empty and done_reason is
"length"; with thinking off the same prompt answers in two tokens and stops.

On frequency, since the first reading of this was wrong: in the research run that
prompted the investigation this fired ONCE for the planner, not the hundred-plus
times a first grep suggested. That count came from the cockpit repainting the same
message every eight seconds — 119 lines, 6 events. The failure mode is real and
worth closing; it was not the reason that run made no progress, and nothing here
should be read as evidence that it was.
"""
import types

import pytest

from spiral.llm import ChatResult, Ollama


def _result(text="", thinking=None, done_reason="stop", **kw):
    return ChatResult(text=text, thinking=thinking, prompt_tokens=kw.get("p", 10),
                      completion_tokens=kw.get("c", 20),
                      raw={"done_reason": done_reason})


# ------------------------------------------------------------------ detection
def test_all_thinking_no_answer_at_the_cap_is_detected():
    assert _result(thinking="Let me work through the cohomology...",
                   done_reason="length").spent_on_thinking


def test_an_ordinary_reply_is_not_mistaken_for_it():
    assert not _result(text="OK").spent_on_thinking
    assert not _result(text="OK", thinking="hmm", done_reason="length").spent_on_thinking


def test_a_genuinely_empty_reply_is_not_a_thinking_overrun():
    """No thinking either — that is a different fault and a retry would not help."""
    assert not _result(done_reason="length").spent_on_thinking
    assert not _result(thinking="  ", done_reason="length").spent_on_thinking


def test_thinking_that_stopped_naturally_is_not_an_overrun():
    """done_reason 'stop' with empty content means the model chose to say nothing;
    the budget is not what stopped it, so buying a second call proves nothing."""
    assert not _result(thinking="pondered", done_reason="stop").spent_on_thinking


# ------------------------------------------------------------------ recovery
class _Fake(Ollama):
    """An Ollama whose transport is a list of canned replies — no HTTP, no model.

    Constructed without super().__init__ on purpose: that opens a real httpx client
    and reads the user's config, neither of which belongs in a unit test.
    """
    def __init__(self, replies):
        self.base_url = "http://x"
        self._no_think = set()
        self.providers = {}
        self.calls = []
        state = {"n": 0}

        def post(url, json=None, **kw):
            self.calls.append(json)
            body = replies[min(state["n"], len(replies) - 1)]
            state["n"] += 1
            return types.SimpleNamespace(
                status_code=200, json=lambda: body, raise_for_status=lambda: None)

        self._client = types.SimpleNamespace(post=post)


THOUGHT_ONLY = {
    "message": {"content": "", "thinking": "Thinking Process:\n1. Analyze..."},
    "done_reason": "length", "prompt_eval_count": 15, "eval_count": 2048,
}
ANSWERED = {
    "message": {"content": '{"angle": "H^5 on S^5"}'},
    "done_reason": "stop", "prompt_eval_count": 15, "eval_count": 40,
}


def test_a_thinking_overrun_is_retried_without_thinking():
    client = _Fake([THOUGHT_ONLY, ANSWERED])

    got = client.chat("qwen3.8:27b", [{"role": "user", "content": "go"}],
                      think=True, num_predict=2048)

    assert got.text == '{"angle": "H^5 on S^5"}', "the retry's answer must come back"
    assert len(client.calls) == 2, "it did not retry"
    assert client.calls[0].get("think") is True
    assert client.calls[1].get("think") is False, "the retry must turn thinking off"
    assert got.raw.get("spiral_recovered_from_thinking") is True


def test_the_wasted_call_is_still_paid_for():
    """Budget accounting must see the tokens the abandoned reasoning really cost,
    or a run silently overspends its ceiling."""
    client = _Fake([THOUGHT_ONLY, ANSWERED])

    got = client.chat("qwen3.8:27b", [{"role": "user", "content": "go"}],
                      think=True, num_predict=2048)

    assert got.completion_tokens == 2048 + 40
    assert got.prompt_tokens == 15 + 15


def test_the_discarded_reasoning_is_kept():
    client = _Fake([THOUGHT_ONLY, ANSWERED])
    got = client.chat("m", [{"role": "user", "content": "go"}], think=True)
    assert "Thinking Process" in (got.thinking or "")


def test_it_retries_at_most_once():
    """Two overruns in a row must not recurse — one wasted call is the ceiling."""
    client = _Fake([THOUGHT_ONLY, THOUGHT_ONLY])

    got = client.chat("m", [{"role": "user", "content": "go"}], think=True)

    assert len(client.calls) == 2, f"retried {len(client.calls) - 1} times"
    assert got.text == ""            # honestly empty; the caller decides what next


def test_a_normal_reply_costs_exactly_one_call():
    client = _Fake([ANSWERED])
    got = client.chat("m", [{"role": "user", "content": "go"}], think=True)
    assert len(client.calls) == 1 and got.text.startswith("{")


def test_thinking_off_is_never_retried():
    """With think=False an empty reply is the model's answer, not a budget wall."""
    client = _Fake([THOUGHT_ONLY])
    got = client.chat("m", [{"role": "user", "content": "go"}], think=False)
    assert len(client.calls) == 1 and got.text == ""


# ------------------------------------------------------------------ live
@pytest.mark.skipif(
    not __import__("shutil").which("ollama"), reason="no local ollama")
def test_against_the_real_model_if_one_is_installed():
    """The bug was found live, so pin it live where the machine allows."""
    import httpx

    client = Ollama()
    try:
        names = [m["name"] for m in
                 httpx.get("http://localhost:11434/api/tags", timeout=5)
                 .json().get("models", [])]
    except Exception:
        pytest.skip("ollama not answering")
    thinker = next((n for n in names if n.startswith("qwen3")), None)
    if not thinker:
        pytest.skip("no thinking model pulled")
    got = client.chat(thinker, [{"role": "user", "content": "Reply with exactly: OK"}],
                      think=True, num_predict=24)
    assert got.text.strip(), "a thinking overrun still came back empty"
