"""Controller base class"""

from quadrotor.dynamics import QuadrotorState, QuadrotorCommands
from quadrotor.trajectory import TrajectoryState


class ControllerBase:
    def step(
        self, t: float, trajectory: TrajectoryState, state: QuadrotorState
    ) -> QuadrotorCommands:
        raise NotImplementedError("Step function not implemented!")
