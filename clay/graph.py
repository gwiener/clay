import dataclasses
from typing import Self


class Node:
    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self.width = width
        self.height = height


@dataclasses.dataclass
class NodeRenderingHints:
    font_name: str = "Arial"
    font_size: int = 12
    font_factor: float = 0.7
    padding: int = 5


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
    
    @classmethod
    def from_node_names(
        cls,
        node_names: list[str],
        edges: list[tuple[str, str]],
        canvas_size: tuple[int, int] = (800, 600),
        defaults: NodeRenderingHints = NodeRenderingHints(),
        padding: int = 5,
    ) -> Self:
        nodes = [
            Node(
                name=name,
                width=defaults.font_size * defaults.font_factor * len(name) + 2 * defaults.padding,
                height=defaults.font_size + 2 * defaults.padding
            )
            for name in node_names
        ]
        return cls(nodes=nodes, edges=edges, canvas_size=canvas_size, defaults=defaults, padding=padding)


class Layout:
    def __init__(self, graph: Graph, centers: list[int]):
        self.graph = graph
        self.centers = centers
    
    def get_node_center(self, node_name: str) -> tuple[int, int]:
        idx = self.graph.name2idx[node_name] * 2
        return (self.centers[idx], self.centers[idx + 1])