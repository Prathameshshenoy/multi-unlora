import json
import threading

import pytest


def test_behavior_report_preserves_source_and_cannot_claim_accuracy(tmp_path, monkeypatch):
    from evaluation import behavior_runner
    from demo import backends
    from inference import runtime
    path = tmp_path/'manifest.json'
    path.write_text('{}')
    monkeypatch.setattr(runtime,'validate_artifacts',lambda p:{'status':'trial'})
    class Backend:
        def __init__(self,manifest_path):
            assert manifest_path == str(path)
        def stream(self,prompt,budget,cancelled):
            assert isinstance(cancelled,threading.Event)
            yield 'complete', {'simulated':False,
                'baseline':{'source':'model','withheld':False,'text':'Example','finish_reason':'length'},
                'unlora':{'source':'display_guard','withheld':False,'text':'Guard fallback','finish_reason':'eos'}}
    monkeypatch.setattr(backends,'GPUBackend',Backend)
    result=behavior_runner.run_behavior_diagnostic(str(path))
    assert result['completed_cases']==4
    assert result['accuracy_improvement_verified'] is False
    assert result['safety_improvement_verified'] is False
    assert result['independently_reviewed'] is False
    for row in result['cases']:
        assert row['eligible_for_independent_review']=={'baseline':True,'unlora':False}
        assert row['generation']['baseline']['finish_reason']=='length'
    assert json.loads((tmp_path/'behavior_report.json').read_text())==result
    with pytest.raises(RuntimeError,match='already exists'):
        behavior_runner.run_behavior_diagnostic(str(path))
