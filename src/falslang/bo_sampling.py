
import time
import numpy as np
from bo.gpr import GPRSkeleton
from bo.sampling import uniform_sampling
from bo.utils import Fn
from numpy.typing import NDArray

from falslang.bo_parallel import ParallelInternalBO


def compute_robustness(samples_in: NDArray[np.float_], test_function: Fn) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Compute the fitness (robustness) of the given sample.

    Args:
        samples_in: Samples points for which the fitness is to be computed.
        test_function: Test Function insitialized with Fn
    Returns:
        Fitness (robustness) of the given sample(s)
    """
    samples_out_dist = []
    samples_out_rtamt = np.empty(samples_in.shape[0])
    if samples_in.shape[0] == 1:
        dist, rtamt = test_function(samples_in[0])
        samples_out_dist = np.array([dist])
        samples_out_rtamt[0] = rtamt

        return samples_out_dist, samples_out_rtamt
    
    for iterate, sample in enumerate(samples_in):
        dist, rtamt = test_function(sample)
        samples_out_dist.append(dist)
        samples_out_rtamt[iterate] = rtamt

    return np.array(samples_out_dist), samples_out_rtamt


class ParallelBOSamplingController:
    """Main controller that handles parallel BO sampling and evaluation"""
    def __init__(
        self,
        test_function: Fn,
        num_boxes: int,
        region_supports: list[NDArray[np.float_]],
        rng: np.random.Generator
    ):
        self.test_function = test_function
        self.num_boxes = num_boxes
        self.region_supports = region_supports
        
        self.rng = rng
        
        # Initialize parallel BO sampler
        self.parallel_bo = ParallelInternalBO(num_boxes)
        
        # Initialize datasets for each box
        self.x_datasets = [np.empty((0, region_support.shape[0])) for region_support in region_supports]
        self.y_datasets = [np.empty(0) for _ in range(num_boxes)]
        self.curr_bests = [None] * num_boxes
        self.times = []
    
    def initialize_datasets(
        self,
        num_init_samples: int,
        initial_datasets: list[tuple[NDArray, NDArray]]|None = None
    ):
        """Initialize or set initial datasets for each box"""
        start = time.perf_counter()
        if initial_datasets is not None:
            # Use provided datasets
            for i, (x_data, y_data) in enumerate(initial_datasets):
                self.x_datasets[i] = x_data
                self.y_datasets[i] = y_data
                self.curr_bests[i] = np.min(y_data) if len(y_data) > 0 else None
        else:
            x_init = uniform_sampling(num_init_samples, self.region_supports, self.region_supports.shape[0], self.rng)
            y_init, _ = compute_robustness(x_init, self.test_function)
            # Generate initial samples using uniform sampling
            for i in range(self.num_boxes):
                # dim = self.region_supports[i].shape[0]
                self.x_datasets[i] = x_init
                self.y_datasets[i] = y_init[:, i]
                self.curr_bests[i] = np.min(y_init)
        self.times.append(time.perf_counter() - start)

    def run_iteration(self) -> tuple[list[NDArray], list[float], list[float | None]]:
        """Run one iteration of parallel BO sampling and evaluation"""
        # Sample new points from all boxes in parallel
        new_points = self.parallel_bo.sample_parallel(
            x_trains=self.x_datasets,
            y_trains=self.y_datasets,
            region_supports=self.region_supports,
            rng=self.rng,
            curr_bests=self.curr_bests
        )
        
        # Evaluate new points in the main process (since compute_robustness is not pickleable)
        evaluation_results = []
        for point in new_points:
            # Your evaluate function returns (cost_boxes, cost_rtamt)
            result = compute_robustness(np.array([point]), self.test_function)
            evaluation_results.append(result)
        
        # Update datasets and check termination
        should_terminate = False
        termination_reason = None
        
        for i, (point, (cost_boxes, cost_rtamt)) in enumerate(zip(new_points, evaluation_results, strict=False)):
            # Update dataset for this box
            self.x_datasets[i] = np.vstack([self.x_datasets[i], point.reshape(1, -1)])
            # Use the appropriate robustness value (adjust based on your evaluate function)
            robustness_value = cost_boxes[0,i]
            self.y_datasets[i] = np.append(self.y_datasets[i], robustness_value)
            self.curr_bests[i] = np.min(self.y_datasets[i])
            
            # Check termination condition for this box
            if cost_rtamt is not None and cost_rtamt < 0 and np.min(cost_boxes) < 0:
                should_terminate = True
                termination_reason = f"Box {i} found satisfactory solution"
            print(f"Box {i}: Box Distance {robustness_value:.3f}\t Robustness {np.min(cost_boxes):.3f} \t RTAMT value {cost_rtamt[0]:.3f} \t Current Best {self.curr_bests[i]:.3f}")
        print("*************************")
        return new_points, evaluation_results, should_terminate, termination_reason
    
    def run_until_termination(
        self,
        max_iterations: int = 100,
        num_init_samples: int = 10,
        initial_datasets: list[tuple[NDArray, NDArray]] | None = None
    ) -> dict:
        """Run BO sampling until termination condition is met or max iterations reached"""
        # Initialize datasets
        self.initialize_datasets(num_init_samples, initial_datasets)
        
        iteration_results = []
        
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}/{max_iterations}")
            start = time.perf_counter()
            new_points, eval_results, should_terminate, reason = self.run_iteration()
            
            # Store iteration results
            iteration_results.append({
                'iteration': iteration,
                'new_points': new_points,
                'evaluation_results': eval_results,
                'datasets': [(x.copy(), y.copy()) for x, y in zip(self.x_datasets, self.y_datasets, strict=False)]
            })
            self.times.append(time.perf_counter() - start)
            # Check termination
            if should_terminate:
                print(f"Termination condition met: {reason}")
                break
        
        return {
            'final_datasets': [(x.copy(), y.copy()) for x, y in zip(self.x_datasets, self.y_datasets, strict=False)],
            'iteration_results': iteration_results,
            'terminated_early': should_terminate,
            'termination_reason': reason if should_terminate else "Max iterations reached",
            'times': self.times
        }