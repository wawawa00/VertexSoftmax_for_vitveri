from __future__ import annotations

from contextlib import contextmanager


CAPTURE_ATTRIBUTE = "_vertex_direct_intermediate_bounds"


def _clone_bound(value):
    value = value.detach() if hasattr(value, "detach") else value
    value = value.cpu() if hasattr(value, "cpu") else value
    return value.clone() if hasattr(value, "clone") else value


def _capture_requested_bounds(net, node_names):
    captured = {}
    for name in dict.fromkeys(node_names):
        node = net[name]
        net.compute_intermediate_bounds(node)
        lower = getattr(node, "lower", None)
        upper = getattr(node, "upper", None)
        if lower is None or upper is None:
            raise RuntimeError(f"AutoLiRPA did not compute both bounds for node {name}")
        if bool((lower > upper).any().item()):
            raise ValueError(f"AutoLiRPA returned inverted intermediate bounds for node {name}")
        captured[name] = (_clone_bound(lower), _clone_bound(upper))
    setattr(net, CAPTURE_ATTRIBUTE, captured)


@contextmanager
def capture_intermediate_bounds(bounded_module_class, node_selector):
    """Capture selected CROWN bounds during init_alpha without editing AutoLiRPA."""
    original = bounded_module_class.init_alpha

    def wrapped(net, *args, **kwargs):
        result = original(net, *args, **kwargs)
        node_names = tuple(node_selector(net))
        if not node_names:
            raise ValueError("The intermediate-bound selector returned no nodes")
        _capture_requested_bounds(net, node_names)
        return result

    bounded_module_class.init_alpha = wrapped
    try:
        yield
    finally:
        bounded_module_class.init_alpha = original
