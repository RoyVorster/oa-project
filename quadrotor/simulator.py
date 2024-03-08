"""Basic simulator implementation"""

import typing as T
from dataclasses import dataclass

import numpy as np

from quadrotor.dynamics import QuadrotorDynamicsBase, QuadrotorState, QuadrotorCommands
from quadrotor.controller import ControllerBase
from quadrotor.trajectory import TrajectoryBase, TrajectoryState


@dataclass
class SimulatorState:
    t: float
    state: QuadrotorState
    command: QuadrotorCommands
    trajectory: TrajectoryState


class SimulatorBase:
    def __init__(
        self,
        dt: float,  # Simulation step time [s]
        dynamics: QuadrotorDynamicsBase,
        controller: ControllerBase,
        trajectory: TrajectoryBase,
        initial_state: T.Optional[QuadrotorState] = None,
        t_total: float = 5.0,  # Simulation duration [s]
    ) -> None:
        self.dynamics = dynamics
        self.controller = controller
        self.trajectory = trajectory

        self.dt = dt
        self.dynamics.dt = dt

        self.t_total = t_total

        # Set the initial state to be the start of the trajectory if None
        if initial_state is None:
            traj = self.trajectory.eval(0.0)
            self.initial_state = QuadrotorState(position=traj.position, velocity=traj.velocity)
        else:
            self.initial_state = initial_state

    def simulate(self) -> T.List[SimulatorState]:
        output: T.List[SimulatorState] = []

        # Set simulation time and state to zero
        t, trajectory = 0.0, self.trajectory.eval(0.0)

        # Initialize the dynamics
        state = self.initial_state
        self.dynamics.reset(state)

        t = 0.0
        while t < self.t_total:
            trajectory = self.trajectory.eval(t)
            command = self.controller.step(t, trajectory, state)
            state = self.dynamics.step(t, command)

            output.append(SimulatorState(t, state, command, trajectory))

            t += self.dt

        return output
