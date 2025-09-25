

import numpy as np
from cars import CCModel
from staliro import Sample, SignalInput, TestOptions

from falslang.monitor import Requirement

boxes_2d  = [
        # Box 1: Event at timestep 1
    (
        [40, -50, -50, -50, -50, -50, -50, -50, -50, -50],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ),
    # Box 2: Event at timestep 2
    (
        [-50, 40, -50, -50, -50, -50, -50, -50, -50, -50],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ),
    # Box 3: Event at timestep 3
    (
        [-50, -50, 40, -50, -50, -50, -50, -50, -50, -50],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ),
    # Box 4: Event at timestep 4
    (
        [-50, -50, -50, 40, -50, -50, -50, -50, -50, -50],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ),
    # Box 5: Event at timestep 5
    (
        [-50, -50, -50, -50, 40, -50, -50, -50, -50, -50],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ),
    # Box 6: Event at timestep 6
    (
        [-50, -50, -50, -50, -50, 40, -50, -50, -50, -50],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ),
    # Box 7: Event at timestep 7
    (
        [-50, -50, -50, -50, -50, -50, 40, -50, -50, -50],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ),
    # Box 8: Event at timestep 8
    (
        [-50, -50, -50, -50, -50, -50, -50, 40, -50, -50],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ),
    # Box 9: Event at timestep 9
    (
        [-50, -50, -50, -50, -50, -50, -50, -50, 40, -50],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ),
    # Box 10: Event at timestep 10
    (
        [-50, -50, -50, -50, -50, -50, -50, -50, -50, 40],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    )
]


model = CCModel(10)

opts = TestOptions(
            runs=1,
            iterations=1,
            tspan=(0, 100),
            signals={
                "throttle": SignalInput(control_points=[(0, 1)] * 5),
                "brake": SignalInput(control_points=[(0, 1)] * 5),
            },
        )    


rng = np.random.default_rng(12345)
sample = Sample(rng.uniform(0,1,10).tolist(), opts)
res = model.simulate(sample)



formula = "not(G (y54>40))"
monitor = Requirement(formula, {"y54": 3}, boxes_2d)
res2 = monitor.evaluate(res.value)


print(res2)