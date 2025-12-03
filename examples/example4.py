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
        ("lookup", "candidates")
    ]
)