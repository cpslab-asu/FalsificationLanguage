# Initialize
import numpy as np

from falslang.interface import BO



import numpy as np
from cars import CCModel
from staliro import Sample, SignalInput, TestOptions, staliro

from falslang.monitor import Requirement


formula = "not(G (y54<40))"
# boxes for the formulas
boxes_2d  = [
        # Box 1: Event at timestep 1
    (
        [40, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ),
    # Box 2: Event at timestep 2
    (
        [-1000, 40, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ),
    # Box 3: Event at timestep 3
    (
        [-1000, -1000, 40, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ),
    # Box 4: Event at timestep 4
    (
        [-1000, -1000, -1000, 40, -1000, -1000, -1000, -1000, -1000, -1000],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ),
    # Box 5: Event at timestep 5
    (
        [-1000, -1000, -1000, -1000, 40, -1000, -1000, -1000, -1000, -1000],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ),
    # Box 6: Event at timestep 6
    (
        [-1000, -1000, -1000, -1000, -1000, 40, -1000, -1000, -1000, -1000],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ),
    # Box 7: Event at timestep 7
    (
        [-1000, -1000, -1000, -1000, -1000, -1000, 40, -1000, -1000, -1000],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ),
    # Box 8: Event at timestep 8
    (
        [-1000, -1000, -1000, -1000, -1000, -1000, -1000, 40, -1000, -1000],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ),
    # Box 9: Event at timestep 9
    (
        [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, 40, -1000],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ),
    # Box 10: Event at timestep 10
    (
        [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, 40],
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    )
]


model = CCModel(10)


monitor = Requirement(formula, {"y54": 3}, boxes_2d)


opts = TestOptions(
        runs=1,
        iterations=100,
        tspan=(0, 100),
        signals={
            "throttle": SignalInput(control_points=[(0, 1)] * 5),
            "brake": SignalInput(control_points=[(0, 1)] * 5),
        },
        seed=12345
)    
    
    

bo = BO(
    init_sampling_budget=20,
    name="Bayesian Optimization",
)

runs = staliro(model, monitor, bo, opts)

print(runs)
print(runs[0].evaluations[0])