import unittest

from vertex_vitveri.direct_crown_hook import (
    CAPTURE_ATTRIBUTE,
    capture_intermediate_bounds,
)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeTensor(self.value)

    def __gt__(self, other):
        return FakeBoolean(self.value > other.value)


class FakeBoolean:
    def __init__(self, value):
        self.value = value

    def any(self):
        return self

    def item(self):
        return self.value


class FakeNode:
    def __init__(self, name):
        self.name = name
        self.lower = None
        self.upper = None


class FakeBoundedModule:
    def __init__(self):
        self.nodes = {"/z": FakeNode("/z"), "/normalized": FakeNode("/normalized")}
        self.calls = []

    def __getitem__(self, name):
        return self.nodes[name]

    def init_alpha(self, marker=None):
        self.calls.append(("init_alpha", marker))
        return "original-result"

    def compute_intermediate_bounds(self, node):
        self.calls.append(("bound", node.name))
        node.lower = FakeTensor(-1)
        node.upper = FakeTensor(2)


class DirectCrownHookTest(unittest.TestCase):
    def test_captures_requested_nodes_and_restores_method(self):
        original = FakeBoundedModule.init_alpha
        net = FakeBoundedModule()

        with capture_intermediate_bounds(
            FakeBoundedModule,
            lambda _net: ("/z", "/normalized"),
        ):
            result = net.init_alpha("marker")

        self.assertEqual(result, "original-result")
        self.assertIs(FakeBoundedModule.init_alpha, original)
        self.assertEqual(
            net.calls,
            [("init_alpha", "marker"), ("bound", "/z"), ("bound", "/normalized")],
        )
        captured = getattr(net, CAPTURE_ATTRIBUTE)
        self.assertEqual(captured["/z"][0].value, -1)
        self.assertEqual(captured["/normalized"][1].value, 2)

    def test_restores_method_when_selector_fails(self):
        original = FakeBoundedModule.init_alpha
        net = FakeBoundedModule()

        with self.assertRaisesRegex(RuntimeError, "selector failed"):
            with capture_intermediate_bounds(
                FakeBoundedModule,
                lambda _net: (_ for _ in ()).throw(RuntimeError("selector failed")),
            ):
                net.init_alpha()

        self.assertIs(FakeBoundedModule.init_alpha, original)


if __name__ == "__main__":
    unittest.main()
