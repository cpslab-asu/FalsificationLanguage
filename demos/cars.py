import pathlib
import time

import matlab
import matlab.engine
import numpy as np
from scipy.interpolate import interp1d
from staliro import Sample
from staliro.models import Model, Result


class CCModel(Model[list[float], float]):
    MODEL_NAME = "cars"

    def __init__(self, num_ts=10) -> None:
        script_path = pathlib.Path(__name__)
        self.sampling_step = 0.2
        self.num_ts = num_ts  # Fixed number of time points for output
        self.engine = matlab.engine.start_matlab()
        self.engine.addpath(str(script_path.parent))
        model_opts = self.engine.simget(CCModel.MODEL_NAME)
        self.model_opts = self.engine.simset(model_opts, "SaveFormat", "Array")

    def simulate(self, sample: Sample) -> Result[list[float], float]:
        assert sample.signals.tspan is not None
        
        tstart, tend = sample.signals.tspan
        duration = tend - tstart
        sim_t = matlab.double([0, tend])
        n_times = duration // self.sampling_step
        signal_times = np.linspace(tstart, tend, num=int(n_times))
        signal_values = np.array(
            [[signal.at_time(t) for t in signal_times] for signal in sample.signals]
        )
        perf_time = time.perf_counter()
        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())
        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )
        
        # Process the simulation results
        data_array = np.array(data)
        y54 = (data_array[:,4]-data_array[:,3]).reshape((-1,1))
        y43 = (data_array[:,3]-data_array[:,2]).reshape((-1,1))
        y32 = (data_array[:,2]-data_array[:,1]).reshape((-1,1))
        y21 = (data_array[:,1]-data_array[:,0]).reshape((-1,1))
        diff_array = np.hstack((y21, y32, y43, y54))
        
        # Get the original simulation timestamps
        original_times = np.array(timestamps).flatten()
        
        # Create fixed time vector with num_ts points
        fixed_times = np.linspace(tstart, tend, self.num_ts)
        
        # Interpolate each state variable to the fixed time grid
        fixed_states = []
        for i in range(diff_array.shape[1]):
            interp_func = interp1d(
                original_times, 
                diff_array[:, i], 
                kind='cubic',  # Using cubic interpolation for smoother results
                bounds_error=False,
                fill_value="extrapolate"
            )
            fixed_states.append(interp_func(fixed_times))
        time_end = time.perf_counter() - perf_time
        # Convert to list format for compatibility
        times_list = fixed_times.tolist()
        states_list = np.array(fixed_states).T.tolist()  # Transpose to match expected format

        return Result(times=times_list, states=states_list, extra=time_end)