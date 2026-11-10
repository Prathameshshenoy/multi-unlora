import pytest
from scripts.reconcile_completed_budget import bounded_reservation


def test_completion_bound_retains_shutdown_rate_and_overhead_margin():
    assert bounded_reservation(1000,1400,1.8)==.51
    assert bounded_reservation(1000,1001,1.8)==.25
    assert bounded_reservation(1000,2700,1.8)==1.8
    with pytest.raises(ValueError): bounded_reservation(1000,999,1.8)
    with pytest.raises(ValueError): bounded_reservation(1000,10000,1.8)
