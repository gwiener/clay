from clay.graph import Graph, Node, Edge

graph = Graph(
    nodes=[Node("foo"), Node("bim"), Node("bar"), Node("baz")],
    edges=[Edge("foo", "bim"), Edge("bim", "bar"), Edge("foo", "baz")]
)
