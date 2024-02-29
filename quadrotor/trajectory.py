"""Trajectory base classes"""

import numpy as np

from dataclasses import dataclass, field


@dataclass
class TrajectoryState:
    # time  [s]
    t: float

    # xyz position  [m]
    position: np.ndarray

    # xyz velocity  [m]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))


class TrajectoryBase:
    def eval(self, t: float) -> TrajectoryState:
        """Evaluate the trajectory at some time t"""
        raise NotImplementedError("")
