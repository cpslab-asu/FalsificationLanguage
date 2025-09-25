from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import ray
import tqdm
from bo.core import BOInterface, BOSampling, InternalBO
from bo.gpr import GPR, GPRSkeleton, InternalGPR
from bo.sampling import uniform_sampling
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm


@ray.remote
class InternalBOWorker:
    """Ray actor for parallel BO sampling"""
    def __init__(self):
        # Recreate the BO model from config (ensure it's pickleable)
        self.bo_model = InternalBO()
    
    def sample(
        self,
        x_train: NDArray[np.float_],
        y_train: NDArray[np.float_],
        region_support: NDArray[np.float_],
        rng_seed: int,
        curr_best: NDArray[np.float_] | None = None
    ) -> NDArray[np.float_]:
        """Sample a new point using BO (runs on worker)"""
        rng = np.random.default_rng(rng_seed)
        
        # Recreate GPR model from config if needed
        # gpr_model = recreate_gpr_from_config(gpr_model_config)
        
        return self.bo_model.sample(
            x_train=x_train,
            y_train=y_train,
            region_support=region_support,
            gpr_model=InternalGPR(),  # Pass config directly if GPR is recreated inside
            rng=rng,
            curr_best=curr_best
        )

class ParallelInternalBO(BOInterface):
    """BO Interface that supports parallel sampling across multiple boxes"""
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.workers = None
        self._init_workers()
    
    def _init_workers(self):
        """Initialize Ray workers"""
        if not ray.is_initialized():
            ray.init()
        
        # Create worker actors
        self.workers = [InternalBOWorker.remote() for _ in range(self.num_workers)]
    
    def sample(
        self,
        x_train: NDArray[np.float_],
        y_train: NDArray[np.float_],
        region_support: NDArray[np.float_],
        gpr_model: GPRSkeleton,
        rng: np.random.Generator,
        curr_best: NDArray[np.float_]|None = None
    ) -> NDArray[np.float_]:
        """Sample a single point (for backward compatibility)"""
        # For single box case, just use the original logic
        bo_model = InternalBO()
        return bo_model.sample(x_train, y_train, region_support, gpr_model, rng, curr_best)
    
    def sample_parallel(
        self,
        x_trains: list[NDArray[np.float_]],
        y_trains: list[NDArray[np.float_]],
        region_supports: list[NDArray[np.float_]],
        rng: np.random.Generator,
        curr_bests: list[NDArray[np.float_]]|None = None
    ) -> list[NDArray[np.float_]]:
        """Sample one point for each box in parallel"""

        if len(x_trains) != self.num_workers:
            raise ValueError(f"Expected {self.num_workers} datasets, got {len(x_trains)}")
        
        if curr_bests is None:
            curr_bests = [None] * self.num_workers
        
        # Submit sampling tasks to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.sample.remote(
                x_train=x_trains[i],
                y_train=y_trains[i],
                region_support=region_supports,
                rng_seed=rng.integers(0, 2**31 - 1),
                curr_best=curr_bests[i]
            )
            futures.append(future)
        
        # Wait for all workers to complete
        return ray.get(futures)