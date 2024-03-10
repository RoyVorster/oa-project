"""Plotting utilities"""

import typing as T
from pathlib import Path

import numpy as np
import k3d
import pyvista as pv

from sym import Rot3

from quadrotor.simulator import SimulatorState


# Skydio X2 model
VTK_MODEL = pv.read(Path(__file__).parent.parent / "assets" / "x2.vtk")


# Translation matrix (i.e. [[R, t], [0, 0, 0, 1]])
def trans_matrix(R: Rot3, t: np.ndarray, scale: float = 100.0) -> np.ndarray:
    R_vtk = Rot3.from_yaw_pitch_roll(0.0, 0.0, np.pi / 2)  # The object is rotated by default
    return np.block([[(R * R_vtk).to_rotation_matrix() / scale, t.reshape(-1, 1)], [np.zeros(3), 1.0 / scale]])


# Animate using k3D
# I expect we'll be expanding this during the course to include things like our reference/actual
# trajectory, etc...
def animate_k3d(output: T.List[SimulatorState]) -> k3d.Plot:
    # Compute limits based on trajectory
    positions = np.array([o.state.position for o in output])
    def get_axis_limits(axis: int) -> T.Tuple[float, float]:
        return [min(positions[:, axis]) - 1.0, max(positions[:, axis]) + 1.0]

    x_limits = get_axis_limits(0)
    y_limits = get_axis_limits(1)
    z_limits = get_axis_limits(2)

    # Initialize plot
    plot = k3d.plot(grid=[x_limits[0], y_limits[0], z_limits[0], x_limits[1], y_limits[1], z_limits[1]], grid_auto_fit=False)

    model = k3d.vtk_poly_data(VTK_MODEL, model_matrix=np.zeros((4, 4)))
    model.model_matrix = {
        str(o.t): trans_matrix(o.state.orientation, o.state.position) for o in output
    }

    plot += model

    return plot


def animate_matplotlib(output: T.List[SimulatorState]) -> None:
    raise NotImplementedError("Not implemented anymore!")