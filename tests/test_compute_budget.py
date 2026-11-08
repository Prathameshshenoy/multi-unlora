import json
import pytest
from scripts import run_engine


def test_short_session_reserves_only_its_predeclared_ceiling(tmp_path,monkeypatch):
    monkeypatch.setattr(run_engine,'ROOT',tmp_path)
    run_engine.reserve('serve-local',minutes=5)
    ledger=json.loads((tmp_path/'.runtime/budget.json').read_text())
    assert ledger['reservations'][-1]['reserved_usd']==.5


def test_budget_refuses_further_work_at_the_ceiling(tmp_path,monkeypatch):
    monkeypatch.setattr(run_engine,'ROOT',tmp_path)
    state=tmp_path/'.runtime'
    state.mkdir()
    (state/'budget.json').write_text(json.dumps({'approved_usd':10,'prior_cpu_preflight_reserve_usd':.5,
        'reservations':[{'stage':'train','reserved_usd':9.4}]}))
    with pytest.raises(RuntimeError,match='ceiling reached'):
        run_engine.reserve('serve-local',minutes=5)
    assert len(json.loads((state/'budget.json').read_text())['reservations'])==1
