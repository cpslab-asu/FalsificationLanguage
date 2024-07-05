from typing import Any

import math
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from staliro.models import State, ode
from staliro.core.sample import Sample
from staliro.options import Options
from staliro.staliro import simulate_model
from staliro.core.model import ExtraResult

from falsi_lang import SpecLangMonitor

# Define the ODE model
@ode()
def nonlinear_model(time: float, state: State, _: Any) -> State:
    x1_dot = state[0] - state[1] + 0.1 * time
    x2_dot = state[1] * math.cos(2 * math.pi * state[0]) + 0.1 * time

    return np.array([x1_dot, x2_dot])



# Define options for psy-taliro. Though this is used for simulating, we will still define it to generate the signals
options = Options(runs=1, iterations=1, interval=(0, 2),  signals=[], static_parameters=[(-1, 1), (-1, 1)])


# Generate bounds from options and generate a sample using the functionality from package
bounds = options.static_parameters
rng = np.random.default_rng(12345)
sample = np.array(Sample([rng.uniform(bound.lower, bound.upper) for bound in bounds]).values)

# # Simulate the model
result = simulate_model(nonlinear_model, options, sample)


# # Plot the signals
# fig, ax1 = plt.subplots(1)
# ax1.plot(result.trace.times, result.trace.states[0], label = r"$\dot{x_1}$")
# ax1.plot(result.trace.times, result.trace.states[1], label = r"$\dot{x_2}$")
# ax1.set_xlabel("Times")
# plt.legend()
# plt.show()

# Initialize the Monitor

phi = r"(always[1,2] (eventually[3,4] (x1 >= 3 and x1 <= 10)) -> (x1 >= 0 and x1 <= 10)) and (always (x1 >= -20 and x1 <= 20))"
pred_map = {"x1":0}
bound = [[-20,20],[-20,20],[-20,20]]
rob_monitor = SpecLangMonitor(specification=phi, predicate_mapping=pred_map, initialBoxes= bound)


# Monitor the generated signal
monitored_value = rob_monitor.evaluate(result.trace.states, result.trace.times)

print(monitored_value)