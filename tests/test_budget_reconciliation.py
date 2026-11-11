import pytest
from scripts.reconcile_completed_budget import bounded_reservation


def test_completion_bound_retains_shutdown_rate_and_overhead_margin():
    assert bounded_reservation(1000,1400,1.8)==.51
    assert bounded_reservation(1000,1001,1.8)==.25
    assert bounded_reservation(1000,2700,1.8)==1.8
    with pytest.raises(ValueError): bounded_reservation(1000,999,1.8)
    with pytest.raises(ValueError): bounded_reservation(1000,10000,1.8)


def test_recovered_completion_requires_same_cloud_execution_and_report_hash(tmp_path):
    import hashlib,json
    from scripts.reconcile_completed_budget import recovered_completion
    report=tmp_path/'content-delta-recovered-quality_report.json';report.write_text('{"status":"not_promoted"}')
    result=tmp_path/'content-delta-result.json';result.write_text('{"evaluation":{"status":"not_promoted"}}')
    recovery={'stage':'content-delta','modal_app_id':'same-app','quality_report_mtime':1700,
        'quality_report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
        'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest()}
    (tmp_path/'content-delta-recovery.json').write_text(json.dumps(recovery))
    assert recovered_completion(tmp_path,'content-delta',{'modal_app_id':'same-app'})==1700
    with pytest.raises(ValueError,match='identity'):
        recovered_completion(tmp_path,'content-delta',{'modal_app_id':'other-app'})
    report.write_text('changed')
    with pytest.raises(ValueError,match='hash'):
        recovered_completion(tmp_path,'content-delta',{'modal_app_id':'same-app'})
