import dataclasses


@dataclasses.dataclass
class NodeRenderingHints:
    font_name: str = "Arial"
    font_size: int = 12
    font_factor: float = 0.7
    padding: int = 5


class Node:
    def __init__(
        self,
        name: str,
        width: int | None = None,
        height: int | None = None,
        hints: NodeRenderingHints | None = None,
    ):
        self.name = name
        self.hints = hints or NodeRenderingHints()
        self.width = width if width is not None else int(
            self.hints.font_size * self.hints.font_factor * len(name) + 2 * self.hints.padding
        )
        self.height = height if height is not None else int(
            self.hints.font_size + 2 * self.hints.padding
        )


@dataclasses.dataclass
class Canvas:
    width: int
    height: int


class Graph:
    def __init__(
        self,
        nodes: list[Node],
        edges: list[tuple[str, str]],
        canvas_size: tuple[int, int] = (800, 600),
        defaults: NodeRenderingHints = NodeRenderingHints(),
        padding: int = 5,
    ):
        self.nodes = nodes
        self.edges = edges
        self.canvas = Canvas(*canvas_size)
        self.defaults = defaults
        self.padding = padding
        self.name2idx = {node.name: idx for idx, node in enumerate(nodes)}
        self.incoming = {node.name: [] for node in nodes}
        self.outgoing = {node.name: [] for node in nodes}
        for src, dst in edges:
            self.outgoing[src].append(dst)
            self.incoming[dst].append(src)


class Layout:
    def __init__(self, graph: Graph, centers: list[int]):
        self.graph = graph
        self.centers = centers

    def get_node_center(self, node_name: str) -> tuple[int, int]:
        idx = self.graph.name2idx[node_name] * 2
        return (self.centers[idx], self.centers[idx + 1])
