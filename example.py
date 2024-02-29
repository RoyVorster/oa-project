"""A very simple example using the provided templates"""

import numpy as np
import matplotlib.pyplot as plt

from quadrotor.dynamics import QuadrotorDynamicsBase, QuadrotorState
from quadrotor.controller import ControllerBase, QuadrotorCommands
from quadrotor.trajectory import TrajectoryBase, TrajectoryState

from quadrotor.simulator import SimulatorBase
from quadrotor.renderer import animate


# Spoofed dynamics
class SpoofedQuadrotorDynamics(QuadrotorDynamicsBase):
    def __init__(self, spoofed_position: np.ndarray = np.zeros(3)) -> None:
        super().__init__()
        self.spoofed_position = spoofed_position

    def step(self, t: float, input: QuadrotorCommands) -> QuadrotorState:
        return QuadrotorState(position=self.spoofed_position)


# Trajectory that just outputs zeros all the time
class UselessTrajectory(TrajectoryBase):
    def eval(self, t: float) -> TrajectoryState:
        return TrajectoryState(t, position=np.zeros(3))


# Controller that does nothing
class UselessController(ControllerBase):
    def __init__(self, rotor_rate: float = 500.0) -> None:
        self.rotor_rates = np.ones(4) * rotor_rate

    def step(self, *args) -> QuadrotorCommands:
        return QuadrotorCommands(np.zeros(4))


if __name__ == "__main__":
    dt = 0.01

    # 1 meter above the ground
    position = np.array([0.0, 0.0, 1.0])
    sim = SimulatorBase(
        dt=0.01,
        dynamics=SpoofedQuadrotorDynamics(position),
        controller=UselessController(),
        trajectory=UselessTrajectory(),
    )

    # Run the simulator
    output = sim.simulate()

    # Render the output
    ani = animate(output)
    plt.show()
