"""Trajectory base classes"""

import numpy as np

from dataclasses import dataclass, field


@dataclass
class TrajectoryState:
    # time  [s]
    t: float

    # xyz position  [m]
    position: np.ndarray

    # xyz velocity  [m / s]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Yaw [rad]
    yaw: float = 0.0

    # Yaw rate [rad / s]
    yaw_rate: float = 0.0


class TrajectoryBase:
    def eval(self, t: float) -> TrajectoryState:
        """Evaluate the trajectory at some time t"""
        raise NotImplementedError("")
