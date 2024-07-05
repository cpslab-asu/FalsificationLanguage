from abc import ABC
from typing import Sequence, Dict, Set
from numpy.typing import NDArray

import numpy as np
from staliro.specifications import RTAMTDense
from staliro.core import Specification
import time
from .box import Box, point
from .LanguageBoxGeneral import FindBox
# This is an example of using a user-defined monitor that takes in a list of 
# requirements and return robustness for all the components. 
# We have four requirements: phi_1, phi_2, phi_3, phi_4
# I need the robustness of individual components.


# class Component:
#     # Defines a single requirement among a list of requirements
#     def __init__(self, identifier, spec, pred_mapping, mapping) -> None:
#         self.id = identifier
#         self.spec = spec
#         self.pred_mapping = pred_mapping
#         self.specification = RTAMTDense(self.spec, self.pred_mapping)
#         self.count = 0
#         self.robustness_history = []
#         self.falsified = False
#         self.io_mapping = mapping
#         self.monitoring_time = []

#     def __call__(self, states, times):
#         self.count += 1
#         start_time = time.perf_counter()
#         robustness = self.specification.evaluate(states, times)
#         self.monitoring_time.append(time.perf_counter() - start_time)
        
#         self.robustness_history.append([self.count, robustness])
#         if robustness < 0:
#             self.falsified = True
#         return robustness

    


class Requirement(Specification[Sequence[float], float], ABC):
    
    def __init__(self, specification: str, predicate_mapping: dict, initialBoxes: Sequence[Sequence[int]]) -> None:
        self.initialBoxes = initialBoxes
        self.tf_dim = len(initialBoxes)
        
        self.specification = RTAMTDense(specification, predicate_mapping)

        self.B = Box(self.tf_dim)
        self.B.Borders = initialBoxes
        self.result = FindBox(self.tf_dim, self.B, self.d1_function, self.d2_function)
        
        # B.Borders = [[-20,20],[-20,20],[-20,20]]
        
    
    def evaluate(self, states: NDArray[np.float_], times: NDArray[np.float_]) -> Dict[int, float]:
        print(states)
        p = point(self.tf_dim)
        p.coord = states[0]
        return self.d1_function(p,self.result)

    def failure_cost(self):
        ...

    def d1_function(self, x,U):
        dist = -100000
        for i in U.Boxes:
            dist = max(dist,i.minDist(x))
        return dist
    
    def d2_function(self, p):
        point_coord = [p.coord]
        point_time = []
        for i in range(0,p.dim):
            point_time.append(i)
        compute_cost = lambda: self.specification.evaluate(point_coord, point_time)
        cost_duration, cost = self._time(compute_cost)
        # print(compute_cost, cost)
        return cost
    
    
    def _time(self, func):
        start_time = time.perf_counter()
        result = func()
        stop_time = time.perf_counter()
        duration = stop_time - start_time

        return duration, result


# Monitor the generated signal
# monitored_value = rob_monitor.evaluate(result.trace.states, result.trace.times)
