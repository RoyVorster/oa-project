"""Dynamics base class"""
from __future__ import annotations

import numpy as np
import sym

from dataclasses import dataclass, field


@dataclass
class QuadrotorState:
    # xyz position  [m]
    position: np.ndarray

    # orientation
    orientation: sym.Rot3 = field(default_factory=sym.Rot3.identity)

    # xyz velocity  [m / s]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # angular velocity (r, p, q)  [rad / s]
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_state_vector(self) -> np.ndarray:
        return np.hstack([
            self.position,
            self.velocity,
            np.array(self.orientation.to_storage()),
            self.angular_velocity
        ])

    @classmethod
    def from_state_vector(cls, state_vector: np.ndarray) -> QuadrotorState:
        return QuadrotorState(
            position=state_vector[:3],
            velocity=state_vector[3:6],
            orientation=sym.Rot3.from_storage(state_vector[6:10]),
            angular_velocity=state_vector[10:],
        )


@dataclass
class QuadrotorCommands:
    # Four rotor rates
    rotor_rates: np.ndarray  # [rad / s]


class QuadrotorDynamicsBase:
    def __init__(self):
        self.dt = 0.01  # Simulation step time [s]

        # Initialize internal state
        self.state = QuadrotorState(position=np.zeros(3))

    def reset(self, state: QuadrotorState) -> None:
        self.state = state

    def step(self, t: float, input: QuadrotorCommands) -> QuadrotorState:
        raise NotImplementedError("Step function not implemented!")

    def set_dt(self, dt: float) -> None:
        self.dt = dt
