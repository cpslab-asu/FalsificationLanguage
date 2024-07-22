import logging
import math
from math import pi
import numpy as np
from collections import OrderedDict

import plotly.graph_objects as go

from staliro.core import worst_eval, worst_run
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model, staliro


from f16Model import F16Model
from staliroBoInterface import Behavior, BO
from bo.gprInterface import InternalGPR
from bo.bayesianOptimization import InternalBO

import pickle

F16_PARAM_MAP = OrderedDict({
    'air_speed': {
        'enabled': False,
        'default': 540
    },
    'angle_of_attack': {
        'enabled': False,
        'default': np.deg2rad(2.1215)
    },
    'angle_of_sideslip': {
        'enabled': False,
        'default': 0
    },
    'roll': {
        'enabled': True,
        'default': None,
        'range': (pi / 4) + np.array((-pi / 20, pi / 30)),
    },
    'pitch': {
        'enabled': True,
        'default': None,
        'range': (-pi / 2) * 0.8 + np.array((0, pi / 20)),
    },
    'yaw': {
        'enabled': True,
        'default': None,
        'range': (-pi / 4) + np.array((-pi / 8, pi / 8)),
    },
    'roll_rate': {
        'enabled': False,
        'default': 0
    },
    'pitch_rate': {
        'enabled': False,
        'default': 0
    },
    'yaw_rate': {
        'enabled': False,
        'default': 0
    },
    'northward_displacement': {
        'enabled': False,
        'default': 0
    },
    'eastward_displacement': {
        'enabled': False,
        'default': 0
    },
    'altitude': {
        'enabled': False,
        'default': 2338.35
    },
    'engine_power_lag': {
        'enabled': False,
        'default': 9
    }
})




phi = "G[0,15](alt>0)"
specification = RTAMTDense(phi, {"alt": 0})

gpr_model = InternalGPR()
bo_model = InternalBO()
optimizer = BO(50, gpr_model, bo_model, "lhs_sampling", Behavior.FALSIFICATION)

initial_conditions = [
    (math.pi / 4) + np.array((-math.pi / 20, math.pi / 30)),
    (-math.pi / 2) * 0.8 + np.array((0, math.pi / 20)),
    (-math.pi / 4) + np.array((-math.pi / 8, math.pi / 8)),
]

options = Options(runs=10, iterations=300, interval=(0, 15), static_parameters=initial_conditions)

f16_model = F16Model(F16_PARAM_MAP)


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    result = staliro(f16_model, specification, optimizer, options)
    best_sample = worst_eval(worst_run(result)).sample
    best_result = simulate_model(f16_model, options, best_sample)
    with open("F16_RTAMT_monitor.pkl", "wb") as f:
        pickle.dump(result, f)