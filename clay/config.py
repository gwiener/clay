from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from clay.graph import Graph
    from clay.penalties import Penalty


class SpacingConfig(BaseModel):
    """Configuration for Spacing penalty."""

    w: float = 1.0
    D: int = 50
    k_edge: float = 1.0
    k_repel: float = 10.0

    def bind(self, g: Graph) -> Penalty:
        from clay.penalties.spacing import Spacing

        return Spacing(g=g, **self.model_dump())


class NodeEdgeConfig(BaseModel):
    """Configuration for NodeEdge penalty."""

    w: float = 2.0

    def bind(self, g: Graph) -> Penalty:
        from clay.penalties.node_edge import NodeEdge

        return NodeEdge(g=g, **self.model_dump())


class EdgeCrossConfig(BaseModel):
    """Configuration for EdgeCross penalty."""

    w: float = 0.5

    def bind(self, g: Graph) -> Penalty:
        from clay.penalties.edge_cross import EgdeCross

        return EgdeCross(g=g, **self.model_dump())


class AreaConfig(BaseModel):
    """Configuration for Area penalty."""

    w: float = 0.5

    def bind(self, g: Graph) -> Penalty:
        from clay.penalties.area import Area

        return Area(g=g, **self.model_dump())


class ChainCollinearityConfig(BaseModel):
    """Configuration for ChainCollinearity penalty."""

    w: float = 0.3

    def bind(self, g: Graph) -> Penalty:
        from clay.penalties.chain_collinearity import ChainCollinearity

        return ChainCollinearity(g=g, **self.model_dump())


class PenaltiesConfig(BaseModel):
    """Container for all penalty configurations."""

    spacing: SpacingConfig = SpacingConfig()
    node_edge: NodeEdgeConfig = NodeEdgeConfig()
    edge_cross: EdgeCrossConfig = EdgeCrossConfig()
    area: AreaConfig = AreaConfig()
    chain_collinearity: ChainCollinearityConfig = ChainCollinearityConfig()

    def bind_all(self, g: Graph) -> list[Penalty]:
        """Bind all penalty configs to a graph, returning penalty instances."""
        return [
            self.spacing.bind(g),
            self.node_edge.bind(g),
            self.edge_cross.bind(g),
            self.area.bind(g),
            self.chain_collinearity.bind(g),
        ]


class CanvasConfig(BaseModel):
    """Canvas configuration."""

    width: int = 800
    height: int = 600
    padding: int = 5


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    method: str = "basinhopping"
    ftol: float = 1e-6
    gtol: float | None = None
    max_iter: int = 2000
    # Basinhopping-specific parameters
    niter: int = 100
    T: float = 1.0
    stepsize: float = 50.0


class GraphInputConfig(BaseModel):
    """Top-level configuration for graph input."""

    canvas: CanvasConfig = CanvasConfig()
    nodes: list[str]
    edges: list[list[str] | dict]
    penalties: PenaltiesConfig = PenaltiesConfig()
    optimizer: OptimizerConfig = OptimizerConfig()

    def build_graph(self) -> Graph:
        """Build a Graph instance from this configuration."""
        from clay.graph import Edge, Graph, Node

        nodes = [Node(name) for name in self.nodes]
        edges = []
        for e in self.edges:
            match e:
                case [src, dst]:
                    edges.append(Edge(src, dst))
                case {"src": src, "dst": dst, **rest}:
                    edges.append(Edge(src=src, dst=dst, **rest))
                case _:
                    raise ValueError(f"Invalid edge format: {e}")

        return Graph(
            nodes,
            edges,
            canvas_size=(self.canvas.width, self.canvas.height),
            padding=self.canvas.padding,
        )


def load_config(path: str | Path) -> GraphInputConfig:
    """Load a GraphInputConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return GraphInputConfig.model_validate(data)
