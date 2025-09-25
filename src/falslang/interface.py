from dataclasses import dataclass
from typing import Any

import numpy as np
from staliro import Sample
from staliro.optimizers import ObjFunc, Optimizer

from .bo_sampling import ParallelBOSamplingController

BOResult = tuple[Any, Any, Any]

# logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())


@dataclass(frozen=False)
class BO(Optimizer[float, BOResult]):
    """The PartX optimizer provides statistical guarantees about the existence of falsifying behaviour in a system."""
    
    init_sampling_budget: int
    name: str

    def optimize(self, func: ObjFunc[float], params: Optimizer.Params) -> BOResult:
        if self.init_sampling_budget > params.budget:
            raise ValueError("Init Sampling budget cannot be greater than Maximmum Budget")
        region_support = np.array(params.input_bounds)
        
        def test_function(sample) -> float:
            return func.eval_sample(sample)
        
        rng = np.random.default_rng(params.seed)
        return ParallelBOSamplingController(
            test_function=test_function,
            num_boxes= len(func._func.spec.initial_boxes),
            region_supports=region_support,
            rng=rng).run_until_termination(
                max_iterations=params.budget,
                num_init_samples=self.init_sampling_budget
            )

       