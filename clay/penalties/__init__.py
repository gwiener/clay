from abc import abstractmethod

import numpy as np

from clay.graph import Graph


class Penalty(object):
    def __init__(
        self,
        g: Graph,
        w: float = 1.0,
    ):
        self.g = g
        self.w = w
    
    @abstractmethod
    def compute(self, centers: np.ndarray) -> float:
        pass

    def __call__(self, centers: np.ndarray) -> float:
        return self.w * self.compute(centers)

