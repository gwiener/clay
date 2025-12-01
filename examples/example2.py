from clay.graph import Graph

graph = Graph.from_node_names(
    node_names=["start", "process", "project", "tasks", "end"],
    edges=[("start", "process"), ("process", "project"), ("project", "tasks"), ("tasks", "end"), ("process", "end")]
)