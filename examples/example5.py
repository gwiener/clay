from clay.graph import Graph

graph = Graph.from_node_names(
    node_names=[
        "input",
        "analyze",
        "content",
        "metadata",
        "online",
        "LLM knowledge",
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
        "tabular",
        "final"
    ],
    edges=[
        ("input", "analyze"),
        ("analyze", "content"),
        ("analyze", "metadata"),
        ("content", "HyDE"),
        ("online", "HyDE"),
        ("LLM knowledge", "HyDE"),
        ("HyDE", "search terms"),
        ("content", "generate"),
        ("generate", "criteria"),
        ("search terms", "lookup"),
        ("index", "lookup"),
        ("tabular", "lookup"),
        ("metadata", "lookup"),
        ("lookup", "candidates"),
        ("candidates", "rerank"),
        ("rerank", "reranked"),
        ("reranked", "filter"),
        ("criteria", "filter"),
        ("filter", "final")
    ]
)