from clay.graph import Graph, Node, Edge

graph = Graph(
    nodes=[Node("start"), Node("process"), Node("project"), Node("tasks"), Node("end")],
    edges=[Edge("start", "process"), Edge("process", "project"), Edge("project", "tasks"), Edge("tasks", "end"), Edge("process", "end")]
)
