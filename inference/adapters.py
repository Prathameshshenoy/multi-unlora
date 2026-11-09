"""One exception-safe boundary for shared mutable PEFT adapter state."""
from contextlib import contextmanager


@contextmanager
def active_adapter(model, adapter=None, *, restore=False):
    """Select a named expert (or clean base), then clean up on every exit.

    Serving uses the default disabled-on-exit contract. ``restore=True`` is for
    callers that intentionally enter with a named adapter active and need that
    state back. Mixed per-layer enable states are restored for real PEFT layers.
    Caller must exclusively own the shared model for the entire generation.
    """
    previous = getattr(model, "active_adapter", None)
    layers = []
    if restore and callable(getattr(model, "modules", None)):
        layers = [(module, module.disable_adapters) for module in model.modules()
                  if isinstance(getattr(module, "disable_adapters", None), bool)
                  and callable(getattr(module, "enable_adapters", None))]
    enabled = getattr(model, "enabled", None)
    try:
        if adapter is None:
            model.disable_adapter_layers()
        else:
            model.enable_adapter_layers()
            model.set_adapter(adapter)
        yield model
    finally:
        if restore and previous is not None:
            model.set_adapter(previous)
        if restore and layers:
            for layer, disabled in layers:
                layer.enable_adapters(not disabled)
        elif restore and enabled is True:
            model.enable_adapter_layers()
        else:
            model.disable_adapter_layers()
