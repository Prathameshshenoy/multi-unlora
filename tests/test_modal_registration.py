"""Validate the actual Modal endpoint decorators without cloud dispatch."""
import time
import pytest

pytest.importorskip('modal')


def test_budgeted_endpoint_registers_without_allocating_compute():
    from modal_engine import register_local_demo
    service = register_local_demo('test-session-token',time.time()+900)
    assert service is not None
