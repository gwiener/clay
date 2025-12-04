from clay import graph
from clay.layout import LayoutEngine, Result, init_random


class Random(LayoutEngine):
    """Random layout engine - places nodes at random positions."""

    def fit(self, g: graph.Graph) -> Result:
        """
        Generate a random layout for the given graph.

        Args:
            g: Graph object containing nodes.

        Returns:
            A Result object with random center positions for each node.
        """
        limits = self.compute_variable_limits(g)
        centers = init_random(g, limits)
        layout = graph.Layout(g, centers)
        return Result(layout)
