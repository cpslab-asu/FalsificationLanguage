import os
import pickle
import time
from abc import ABC
from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from staliro import Result, Trace
from staliro.specifications import Specification
from staliro.specifications.rtamt import parse_discrete


class Requirement(Specification[Sequence[float], float, None]):
    
    def __init__(self, specification: str, predicate_mapping: dict, initial_boxes: list[tuple[list[float], list[float]]]) -> None:
        self.initial_boxes = [([float(x) for x in box[0]], [float(x) for x in box[1]]) for box in initial_boxes]
        self.tf_dim = len(initial_boxes)
        self.spec_str = specification
        self.predicate_mapping = predicate_mapping
        self.specification = parse_discrete(specification, predicate_mapping)

        # self.time_history = []
        self.output_history = []

    
    def evaluate(self, res: Trace[Sequence[float]]) -> Result[tuple[list[float],float|None], None]:
        
        # if cost_rtamt <= 0:
        #     print(f"RTAMT found falsification with robustness {cost_rtamt}")
        start_time = time.perf_counter()
        cost_boxes = self.distance(np.array(res.states))
        cost_rtamt = -1*self.specification.evaluate(res).value if np.min(cost_boxes) <= 0 else None
        end_time = time.perf_counter() - start_time

        return Result(value = (cost_boxes, cost_rtamt), extra=end_time)
    
    def distance(self, state: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Compute the distance from the current state to the nearest box in the initial_boxes list.

        Args:
            state (NDArray[np.float_]): The current state vector.
        """
        distances = []
        for lower, upper in self.initial_boxes:
            lower = np.array(lower)
            upper = np.array(upper)
            # Compute the distance to the box
            delta = np.min(np.column_stack([state[:,3] - lower, upper - state[:,3]]))
            distances.append(-1*delta)
        return np.array(distances)

    