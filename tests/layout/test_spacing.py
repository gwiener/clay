import numpy as np

from clay.graph import Graph, Node
from clay.layout import Spacing

g = Graph(
    nodes=[Node("a", 10, 10), Node("b", 10, 10)],
    edges=[]
)

func = Spacing(g)
print(func(np.array([20.0, 20.0, 22.0, 22.0])))  # very near
print(func(np.array([20.0, 20.0, 30.0, 30.0])))  # near
print(func(np.array([20.0, 20.0, 50.0, 50.0])))  # farer
