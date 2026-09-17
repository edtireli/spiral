"""Visual work follows the reviewed declaration, including nonvisual documents."""
import json
from types import SimpleNamespace
import pytest

from spiral.conductor import Conductor
from spiral.planner import DeliverableManifestError


def runner(tmp_path, rows, *, goal="requested change", saved_goal=None):
    obj=object.__new__(Conductor);obj.ws=tmp_path
    obj.cfg=SimpleNamespace(visual_review=False)
    obj.state={};obj._write_state=lambda **values:obj.state.update(values)
    obj._dir().joinpath('artifacts.json').write_text(json.dumps({
        'goal_sha256':obj._goal_hash(goal if saved_goal is None else saved_goal),
        'primary_id':rows[0]['id'] if rows else '', 'deliverables':rows}))
    return obj


def test_nonvisual_document_does_not_start_visual_review_or_design(tmp_path):
    obj=runner(tmp_path,[{'id':'readme','kind':'document','visual':False}])
    assert obj._visual_deliverable_targets('requested change')=={}
    obj._visual_review_loop('requested change',None,None)
    assert obj.state['visual_review']=='not-applicable'
    assert obj.state['visual_reviews']=={}


def test_all_declared_visual_deliverables_remain_obligations(tmp_path):
    obj=runner(tmp_path,[{'id':'service','kind':'service','visual':False},
                        {'id':'report','kind':'document','visual':True},
                        {'id':'chart','kind':'plot','visual':True}])
    assert obj._visual_deliverable_targets('requested change')=={'document':['report'],'plot':['chart']}
    obj._visual_review_loop('requested change',None,SimpleNamespace(print=lambda *a:None))
    assert obj.state['visual_reviews']=={'report':'disabled-by-user','chart':'disabled-by-user'}
    assert obj.state['visual_review']=='disabled-by-user'


@pytest.mark.parametrize('visual',[None,0,'false'])
def test_missing_or_untyped_visual_scope_is_not_a_nonvisual_success(tmp_path,visual):
    obj=runner(tmp_path,[{'id':'report','kind':'document','visual':visual}])
    with pytest.raises(DeliverableManifestError):obj._visual_review_loop('requested change',None,None)
    assert obj.state=={}


def test_stale_or_missing_declaration_cannot_invent_delivery_or_visual_scope(tmp_path):
    obj=runner(tmp_path,[{'id':'readme','kind':'document','visual':False}],saved_goal='old task')
    obj.gate_disp=''
    for remove in (False,True):
        if remove: (obj._dir()/'artifacts.json').unlink()
        with pytest.raises(DeliverableManifestError):obj._visual_deliverable_targets('requested change')
        delivery=obj._delivery_manifest('requested change')
        assert delivery['ready'] is False and delivery['deliverables']==[]
        assert 'no inferred substitute' in delivery['declaration_error']
    assert json.loads((obj._dir()/'delivery.json').read_text())['ready'] is False


def test_delivery_honors_nonvisual_text_but_keeps_declared_visual_evidence_required(tmp_path):
    from spiral.delivery import build_delivery_manifest
    (tmp_path/'README.md').write_text('# Actual usage\n\nRun the program with a local input file.\n')
    row={'id':'readme','kind':'document','visual':False,'output_globs':['README.md']}
    declaration={'deliverables':[row]}
    text=build_delivery_manifest(tmp_path,declaration)
    assert text['ready'] is True
    assert text['deliverables'][0]['visual_status']=='not-applicable'
    row['visual']=True
    visual=build_delivery_manifest(tmp_path,declaration)
    assert visual['ready'] is False and visual['deliverables'][0]['visual_required'] is True
    assert build_delivery_manifest(tmp_path,declaration,visual_status='green')['ready'] is True


@pytest.mark.parametrize('budget',[False,True])
def test_unavailable_analysis_cannot_create_a_guessed_contract(tmp_path,monkeypatch,budget):
    from spiral import conductor
    from spiral.execution import BudgetExceeded
    from test_joint_analysis import runner_fixture
    from test_planner import _PlannerModels,_reply
    obj=runner_fixture(tmp_path,monkeypatch,_PlannerModels([]))
    obj.cfg.planning_analysis_mode='sequential'
    monkeypatch.setattr(conductor,'extract_spec',lambda *a,**k:([{'id':'R1','text':'the exact task'}],_reply('{}')))
    error=BudgetExceeded('wall',{}) if budget else RuntimeError('analyst unavailable')
    def unavailable(*a,**k):raise error
    monkeypatch.setattr(conductor,'analyze_deliverables',unavailable)
    with pytest.raises(BudgetExceeded if budget else DeliverableManifestError) as caught:
        obj.make_plan('Repair documentation for an existing command-line interface.')
    if budget:assert caught.value is error
    assert not (obj._dir()/'artifacts.json').exists()
