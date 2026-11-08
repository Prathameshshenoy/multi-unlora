"""Validate the actual Modal endpoint decorators without cloud dispatch."""
import time
import pytest

pytest.importorskip('modal')


def test_budgeted_endpoint_registers_without_allocating_compute():
    from modal_engine import register_local_demo
    service = register_local_demo('test-session-token',time.time()+900)
    assert service is not None


def test_configurable_runtime_base_has_no_local_source_mounts():
    from modal_engine import base_image
    # This is the SDK's own build-order invariant, evaluated without network.
    base_image._assert_no_mount_layers()
    base_image.env({'MULTI_UNLORA_BACKEND':'gpu'})._assert_no_mount_layers()
