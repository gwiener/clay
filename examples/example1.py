from clay.graph import Graph

graph = Graph.from_node_names(
    node_names=["foo", "bim", "bar", "baz"],
    edges=[("foo", "bim"), ("bim", "bar"), ("foo", "baz")]
)