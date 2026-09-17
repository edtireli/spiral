"""Planning recovery revalidates public outputs without spending inference twice."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from spiral.execution import BudgetExceeded, BudgetLimits, RunBudget
from spiral.llm import ChatResult
from spiral.planner import extract_spec
from spiral.planning_memory import PlanningStage


class Model:
    providers = {}

    def __init__(self, *, binding="gguf-v1", text=None):
        self.binding = binding
        self.calls = 0
        self.budget = RunBudget(BudgetLimits(60, 10000, 5))
        self.text = text or json.dumps({"requirements": [
            {"id": "R1", "text": "Preserve the supplied assertions.", "kind": "constraint"}]})

    def source_tokenizer(self, model):
        if self.binding is None:
            return None
        return SimpleNamespace(model=model, inference_identity=self.binding, _check=lambda: None)

    def chat(self, *args, **kwargs):
        self.calls += 1
        return ChatResult(self.text, 100, 30, thinking="private trace must not persist",
                          raw={"done_reason": "stop", "irrelevant_secret": "do not store"})


class Config:
    planner = SimpleNamespace(name="chosen:exact")
    planner_max_tokens = 4096
    keep_alive = "5m"

    def spec_for(self, name):
        return SimpleNamespace(num_ctx=8192)


def stage(root, model, **kwargs):
    return PlanningStage(root, "spec", model, implementation="planner-v1", **kwargs)


def test_resume_revalidates_completed_stage_after_later_interruption(tmp_path):
    first = Model()
    with stage(tmp_path, first) as io:
        spec, _ = extract_spec("Keep the tests", Config(), io)
    # A later stage never returned; it must not become a durable response.
    with pytest.raises(KeyboardInterrupt):
        with PlanningStage(tmp_path, "draft", first, implementation="planner-v1") as io:
            io.chat("chosen:exact", [{"role": "user", "content": "draft"}])
            raise KeyboardInterrupt()
    assert not (tmp_path / ".spiral/planning-checkpoints/draft.json").exists()
    second, updates = Model(), []
    with stage(tmp_path, second, replay=True) as io:
        restored, result = extract_spec("Keep the tests", Config(), io,
                                       progress=lambda *args: updates.append(args))
    assert restored == spec
    assert second.calls == 0 and result.total_tokens == 0 and updates == []
    assert result.raw["planning_checkpoint"]["original_tokens"] == {"prompt": 100, "completion": 30}
    saved = (tmp_path / ".spiral/planning-checkpoints/spec.json").read_text()
    assert "private trace" not in saved and "irrelevant_secret" not in saved


@pytest.mark.parametrize("change", ["request", "model", "binding", "context", "schema_order", "implementation", "fresh"])
def test_changed_input_or_identity_never_reuses_a_response(tmp_path, change):
    options = {"num_ctx": 8192, "fmt": {"properties": {"call": {}, "contract": {}}}}
    messages = [{"role": "user", "content": "same"}]
    with stage(tmp_path, Model()) as io:
        io.chat("chosen:exact", messages, **options)
    model = Model(binding="gguf-v2" if change == "binding" else "gguf-v1")
    if change == "request":
        messages = [{"role": "user", "content": "new constraint"}]
    if change == "context":
        options["num_ctx"] = 32768
    if change == "schema_order":
        options["fmt"] = {"properties": {"contract": {}, "call": {}}}
    with PlanningStage(tmp_path, "spec", model, replay=change != "fresh",
                       implementation="planner-v2" if change == "implementation" else "planner-v1") as io:
        io.chat("different:model" if change == "model" else "chosen:exact", messages, **options)
    assert model.calls == 1


def test_invalid_contract_is_not_checkpointed_and_cached_outputs_are_revalidated(tmp_path, monkeypatch):
    invalid = Model(text='{"requirements": []}')
    with pytest.raises(ValueError, match="non-empty"):
        with stage(tmp_path, invalid) as io:
            extract_spec("Keep the tests", Config(), io)
    assert not (tmp_path / ".spiral/planning-checkpoints/spec.json").exists()
    with stage(tmp_path, Model()) as io:
        extract_spec("Keep the tests", Config(), io)
    from spiral import planner
    monkeypatch.setattr(planner, "_requirements_from_data",
                        lambda data: (_ for _ in ()).throw(ValueError("new validator rejects")))
    second = Model()
    with pytest.raises(ValueError, match="new validator"):
        with stage(tmp_path, second, replay=True) as io:
            extract_spec("Keep the tests", Config(), io)
    assert second.calls == 0


def test_failed_public_proposal_is_archived_without_becoming_resume_authority(tmp_path):
    events=[]
    with pytest.raises(ValueError, match="rejected contract"):
        with stage(tmp_path, Model(text='{"proposal":"incomplete"}'), on_event=lambda **e:events.append(e)) as io:
            io.chat("chosen:exact", [])
            raise ValueError("rejected contract")
    failure=next((tmp_path/'.spiral/planning-failures').glob('spec-*.json'))
    raw=failure.read_text();saved=json.loads(raw)
    assert saved['reusable'] is False and saved['error']=='rejected contract'
    assert saved['entries'][0]['reply']['text']=='{"proposal":"incomplete"}'
    assert 'private trace' not in raw and 'irrelevant_secret' not in raw
    assert events[-1]['reusable'] is False
    assert not (tmp_path/'.spiral/planning-checkpoints/spec.json').exists()
    second=Model()
    with stage(tmp_path,second,replay=True) as io:io.chat("chosen:exact", [])
    assert second.calls==1 and failure.is_file()


def test_failure_archiving_cannot_escape_or_mask_original_error(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (workspace/'.spiral').mkdir()
    (workspace/'.spiral/planning-failures').symlink_to(outside,target_is_directory=True)
    events=[];error=ValueError('original scope error')
    with pytest.raises(ValueError) as caught:
        with stage(workspace,Model(),on_event=lambda **e:events.append(e)) as io:
            io.chat('chosen:exact',[])
            raise error
    assert caught.value is error and list(outside.iterdir())==[]
    assert events[-1]['outcome']=='failure_archive_unavailable'


def test_failure_archive_observer_error_does_not_mask_original_rejection(tmp_path):
    def unavailable(**event): raise RuntimeError('diagnostic observer failed')
    original=ValueError('contract rejected')
    with pytest.raises(ValueError) as caught:
        with stage(tmp_path,Model(),on_event=unavailable) as io:
            io.chat('chosen:exact',[])
            raise original
    assert caught.value is original


def test_mutable_tag_or_corrupt_checkpoint_requires_real_call(tmp_path):
    with stage(tmp_path, Model()) as io:
        io.chat("chosen:exact", [])
    unbound = Model(binding=None)
    with stage(tmp_path, unbound, replay=True) as io:
        io.chat("chosen:exact", [])
    assert unbound.calls == 1
    target = tmp_path / ".spiral/planning-checkpoints/spec.json"
    target.write_text('{"version": 1, "entries": []}')
    model = Model()
    with stage(tmp_path, model, replay=True) as io:
        io.chat("chosen:exact", [])
    assert model.calls == 1


def test_replay_cannot_extend_an_exhausted_budget(tmp_path):
    with stage(tmp_path, Model()) as io:
        io.chat("chosen:exact", [])
    model = Model()
    model.budget.calls = model.budget.limits.model_calls
    with pytest.raises(BudgetExceeded):
        with stage(tmp_path, model, replay=True) as io:
            io.chat("chosen:exact", [])
    assert model.calls == 0


def test_interruption_during_binding_check_is_not_swallowed(tmp_path):
    model = Model()
    def interrupted(_model):
        raise RuntimeError("managed Stop requested")
    model.source_tokenizer = interrupted
    with pytest.raises(RuntimeError, match="Stop requested"):
        with stage(tmp_path, model, replay=True) as io:
            io.chat("chosen:exact", [])
    assert model.calls == 0


def test_stage_storage_cannot_escape_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / ".spiral").symlink_to(outside, target_is_directory=True)
    with stage(workspace, Model()) as io:
        io.chat("chosen:exact", [])
    assert list(outside.iterdir()) == []


def test_new_process_recovers_conductor_stage_with_no_model_call(tmp_path):
    program = '''
import json, sys
from pathlib import Path
from types import SimpleNamespace
from test_planning_memory import Model, Config
from spiral.conductor import Conductor
from spiral.planner import extract_spec
r = object.__new__(Conductor)
r.ws = Path(sys.argv[1]); r.ol = Model()
r.ledger = SimpleNamespace(log=lambda *a, **k: None)
r._resume_planning = sys.argv[2] == "resume"
with r._planning_stage("spec") as io:
    spec, result = extract_spec("Preserve assertions", Config(), io)
print(json.dumps({"calls": r.ol.calls, "spec": spec, "tokens": result.total_tokens}))
'''
    env = dict(os.environ, SPIRAL_OFFLINE_TESTS="1", PYTHONPATH=os.pathsep.join([
        str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]))
    env.pop("SPIRAL_UI_EVENT_TOKEN", None)
    outcomes = []
    for mode in ("fresh", "resume"):
        run = subprocess.run([sys.executable, "-c", program, str(tmp_path), mode],
                             env=env, capture_output=True, text=True, timeout=15, check=True)
        outcomes.append(json.loads(run.stdout))
    assert [row["calls"] for row in outcomes] == [1, 0]
    assert outcomes[0]["spec"] == outcomes[1]["spec"]
    assert outcomes[1]["tokens"] == 0
