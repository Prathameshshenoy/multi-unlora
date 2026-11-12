from types import SimpleNamespace
import pytest
from scripts.run_engine import dispatch_stage,RESERVATIONS

CASES=[('prepare-local','prepare_demo_runtime',()),('adversarial-dev','develop_adversarial_safety',()),('safety-dev','develop_raw_safety',()),('constraint-audit','audit_shared_constraint',()),('selective-confirm','confirm_selective_subtraction',()),('selective-dev','develop_selective_subtraction',()),('support-confirm','confirm_reference_support',()),('support-dev','develop_reference_support',()),('divergence-audit','audit_factual_divergence',()),('strength-dev','calibrate_factual_strength',()),('reference-dev','develop_factual_reference',()),('freeform-dev','develop_freeform',()),('content-delta','repair_content_delta',()),('role-repair','repair_roles',()),('negative-train','train_negative',()),('negative-evaluate','evaluate_negative',()),('matched-train','train_matched',()),('matched-evaluate','evaluate_matched',()),('preflight','preflight',()),('data-audit','audit_training_data',()),('train','train_pilot',(80,)),
       ('evaluate','evaluate_pilot',()),('tune-confirm','tune_and_confirm',()),
       ('paired-train','train_paired',(80,)),('paired-evaluate','evaluate_paired',()),
       ('anchored-train','train_anchored',(80,)),('anchored-evaluate','evaluate_anchored',()),
       ('anchored-long-train','train_anchored_long',()),('anchored-long-evaluate','evaluate_anchored_long',()),
       ('short-answers','evaluate_short_answers',()),('behavior','evaluate_behavior',())]


@pytest.mark.parametrize('stage,function,args',CASES)
def test_each_reservation_dispatches_exactly_its_one_cloud_job(stage,function,args):
    calls=[]
    def job(name):
        def remote(*passed):
            calls.append((name,passed));return {'completed':name}
        return SimpleNamespace(remote=remote)
    engine=SimpleNamespace(**{name:job(name) for _,name,_ in CASES})
    assert dispatch_stage(engine,stage,80)=={'completed':function}
    assert calls==[(function,args)]


def test_every_nonserving_budget_stage_has_a_dispatch_contract():
    assert {case[0] for case in CASES}==set(RESERVATIONS)-{'serve-local'}


def test_model_preparation_uses_the_selected_version():
    calls=[]
    engine=SimpleNamespace(**{name:SimpleNamespace(remote=lambda *args:calls.append(args)) for _,name,_ in CASES})
    dispatch_stage(engine,'prepare-local',80,manifest_path='/vol/runs/version_2/manifest.json')
    assert calls==[('/vol/runs/version_2/manifest.json',)]
