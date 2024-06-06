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

from falsi_lang import ComponentWiseRequirement

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

# Simulate the model
result = simulate_model(nonlinear_model, options, sample)


# Plot the signals
fig, ax1 = plt.subplots(1)
ax1.plot(result.trace.times, result.trace.states[0], label = r"$\dot{x_1}$")
ax1.plot(result.trace.times, result.trace.states[1], label = r"$\dot{x_2}$")
ax1.set_xlabel("Times")
plt.legend()
plt.show()

# Define the different requirements and put them in a list
phi_1 = "always !(x_dot_1 >= -1.6 and x_dot_1 <= -1.4  and x_dot_2 >= -1.1 and x_dot_2 <= -0.9)"
phi_2 = "always !(x_dot_1 >= -1.3 and x_dot_1 <= -1.3  and x_dot_2 >= -1.2 and x_dot_2 <= -0.2)"
phi_3 = "eventually !(x_dot_1 >= -1.6 and x_dot_1 <= -1.4  and x_dot_2 >= -1.1 and x_dot_2 <= -0.9)"
phi_4 = "eventually !(x_dot_1 >= -1.3 and x_dot_1 <= -1.3  and x_dot_2 >= -1.2 and x_dot_2 <= -0.2)"
phi_list = [phi_1, phi_2, phi_3, phi_4]

# Map the predicates in the singals to inputs and outputs. 
# x_dot_1 depends on 0th and 1st input - hence first value of tuple is [0,1]
# x_dot_1 is 0th index singal in the output - hence second value of tuple is 0
# x_dot_2 depends on 0th and 1st input - hence first value of tuple is [0,1]
# x_dot_2 is 1st index singal in the output - hence second value of tuple is 1
predicate_map = {"x_dot_1": ([0,1], 0), "x_dot_2": ([0,1], 1)}

# Initialize the Monitor

input_dimensionality = 2
specification = ComponentWiseRequirement(
                    tf_dim=input_dimensionality, 
                    component_list=phi_list,
                    predicate_mapping=predicate_map
                )

# Monitor the generated signal
monitored_value = specification.evaluate(result.trace.states, result.trace.times)

print(monitored_value)