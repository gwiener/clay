from clay.graph import Graph, Node, Edge

graph = Graph(
    nodes=[
        Node("input"),
        Node("analyze"),
        Node("content"),
        Node("metadata"),
        Node("HyDE"),
        Node("search terms"),
        Node("generate"),
        Node("criteria"),
        Node("index"),
        Node("lookup"),
        Node("candidates"),
    ],
    edges=[
        Edge("input", "analyze"),
        Edge("analyze", "content"),
        Edge("analyze", "metadata"),
        Edge("content", "HyDE"),
        Edge("HyDE", "search terms"),
        Edge("content", "generate"),
        Edge("generate", "criteria"),
        Edge("search terms", "lookup"),
        Edge("index", "lookup"),
        Edge("lookup", "candidates"),
    ]
)
