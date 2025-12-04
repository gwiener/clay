from clay.graph import Graph

graph = Graph.from_node_names(
    node_names=[
        "input",
        "analyze",
        "content",
        "metadata",
        "HyDE",
        "search terms",
        "generate",
        "criteria",
        "index",
        "lookup",
        "candidates",
        "rerank",
        "reranked",
        "filter",
        "final"
    ],
    edges=[
        ("input", "analyze"),
        ("analyze", "content"),
        ("analyze", "metadata"),
        ("content", "HyDE"),
        ("HyDE", "search terms"),
        ("content", "generate"),
        ("generate", "criteria"),
        ("search terms", "lookup"),
        ("index", "lookup"),
        ("lookup", "candidates"),
        ("candidates", "rerank"),
        ("rerank", "reranked"),
        ("reranked", "filter"),
        ("criteria", "filter"),
        ("filter", "final")
    ]
)