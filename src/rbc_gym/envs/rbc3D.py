import logging
from pathlib import Path
import warnings
import os
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import matplotlib

import gymnasium as gym
import juliacall
import juliapkg
import numpy as np

# Optional dependency for 3‑D rendering
try:
    import pyvista as pv

except ImportError:  # pragma: no cover
    pv = None


class RBC3DField(IntEnum):
    T = 0
    U = 1
    V = 2
    W = 3
    P_HY = 4
    P_NHS = 5


def colormap(value, vmin=1, vmax=2, colormap="turbo"):
    cmap = matplotlib.colormaps[colormap]
    value = (value - vmin) / (vmax - vmin)
    return cmap(value, bytes=True)[:, :, :3]


class RayleighBenardConvection3DEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        rayleigh_number: Optional[int] = 2500,
        prandtl_number: Optional[float] = 0.7,
        domain: Optional[list] = [2, 4 * np.pi, 4 * np.pi],
        state_shape: Optional[list] = (16, 32, 32),
        temperature_difference: Optional[list] = [1, 2],
        heater_segments: Optional[int] = 8,
        heater_limit: Optional[float] = 0.9,
        heater_duration: Optional[float] = 0.125,
        episode_length: Optional[int] = 300,
        dt_solver: Optional[float] = 0.01,
        pressure: Optional[bool] = False,
        use_gpu: Optional[bool] = False,
        checkpoint: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the Rayleigh-Benard environment with the given configuration Dictionary.
        """
        super().__init__()
        self.closed = False
        self.use_gpu = use_gpu
        self.checkpoint = checkpoint

        # Environment configuration
        self.ra = rayleigh_number
        self.pr = prandtl_number
        self.domain = domain
        self.episode_length = episode_length
        self.dt_solver = dt_solver
        self.state_shape = state_shape
        self.temperature_difference = temperature_difference
        self.heater_segments = heater_segments
        self.heater_limit = heater_limit
        self.heater_duration = heater_duration
        self.include_pressure = pressure

        # Print environment configuration
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using Rayleigh number Ra={self.ra}")
        self.logger.info(f"Using episode length {self.episode_length} timesteps")

        # The agent takes actions between [-1, 1] on the bottom segments
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(self.heater_segments, self.heater_segments), dtype=np.float32
        )

        # Observation Space
        channels = 4
        lows = [
            np.full(self.state_shape, self.temperature_difference[0]),
            np.full(self.state_shape, -np.inf),
            np.full(self.state_shape, -np.inf),
            np.full(self.state_shape, -np.inf),
        ]
        highs = [
            np.full(
                self.state_shape, self.temperature_difference[1] + self.heater_limit
            ),
            np.full(self.state_shape, np.inf),
            np.full(self.state_shape, np.inf),
            np.full(self.state_shape, np.inf),
        ]

        if self.include_pressure:
            channels += 2
            lows.append(np.full(self.state_shape, -np.inf))
            lows.append(np.full(self.state_shape, -np.inf))
            highs.append(np.full(self.state_shape, np.inf))
            highs.append(np.full(self.state_shape, np.inf))

        self.observation_space = gym.spaces.Box(
            np.stack(lows, dtype=np.float32, axis=0),
            np.stack(highs, dtype=np.float32, axis=0),
            shape=(
                channels,
                self.state_shape[0],
                self.state_shape[1],
                self.state_shape[2],
            ),
            dtype=np.float32,
        )

        # Julia simulation
        juliapkg.resolve()
        self.sim = juliacall.newmodule("RBCGymAPI")
        package_dir = os.path.dirname(os.path.abspath(__file__))
        julia_file_path = os.path.join(package_dir, "..", "sim", "rbc_sim3D_api.jl")
        self.sim.include(julia_file_path)

        # Rendering
        self.render_mode = render_mode
        self._plotter = None
        self._grid = None
        self._volume = None

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)
        # Set checkpoint
        path = None
        if self.checkpoint:
            path = Path(self.checkpoint)
            self.logger.info(f"Using checkpoint file {path.absolute()}")
            if not path.exists():
                raise FileNotFoundError(
                    f"Checkpoint file {path} does not exist. "
                    "Please provide a valid checkpoint directory."
                )
            path = str(path.absolute())
        # initialize julia simulation
        self.sim.initialize_simulation(
            Ra=self.ra,
            Pr=self.pr,
            L=self.domain[::-1],
            grid=self.state_shape[::-1],  # julia uses column-major order
            T_diff=self.temperature_difference,
            heaters=self.heater_segments,
            heater_limit=self.heater_limit,
            dt=self.heater_duration,
            dt_solver=self.dt_solver,
            seed=self.np_random_seed,
            checkpoint_path=path,
            use_gpu=self.use_gpu,
        )

        # Reset action
        self.last_action = self.action_space.sample() * 0

        return self.__get_obs(), self.__get_info()

    def step(self, action: Any = None) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        terminated = False  # is always false; no terminal state
        truncated = False
        # zero action if none
        if action is None:
            action = np.zeros(self.action_space.shape, dtype=np.float32)
            warnings.warn("No action provided, using zero action")

        # Simulation Step
        success = self.sim.step_simulation(np.array(action))
        if not success:
            raise RuntimeError("Error in simulation step, probably NaN values")

        # Get current info
        self.last_obs = self.__get_obs()
        self.last_reward = self.__get_reward()
        self.last_info = self.__get_info()

        # Check for truncation
        if self.last_info["t"] >= self.episode_length:
            truncated = True

        return self.last_obs, self.last_reward, terminated, truncated, self.last_info

    def __get_obs(self) -> Any:
        obs = np.array(self.sim.get_state(), dtype=np.float32)
        if not self.include_pressure:
            obs = obs[:4, :, :]
        obs = obs.transpose(0, 3, 2, 1)  # julia uses column-major order
        return obs

    def __get_reward(self) -> float:
        nu = self.sim.get_nusselt()
        return -nu

    def __get_info(self) -> dict[str, Any]:
        t, step = self.sim.get_info()
        nu = self.sim.get_nusselt()
        return {
            "t": t,
            "step": step,
            "nusselt": nu,
        }

    def render(self):
        """Visualise the current 3-D temperature field using PyVista."""
        if self.render_mode == "human":
            live = True
        elif self.render_mode == "rgb_array":
            live = False
        else:
            return None

        if pv is None:
            raise ModuleNotFoundError(
                "PyVista is required for volume rendering. "
                "Install it with `pip install pyvista`."
            )

        # Temperature is channel 0
        T = self.__get_obs()[RBC3DField.T]
        T = np.flip(T, axis=1)

        # get min max value
        cmin = self.temperature_difference[0]
        cmax = self.temperature_difference[1]

        if self._grid is None:
            # Build a rectilinear grid (VTK’s preferred format for cell‑centred data)
            nz, ny, nx = T.shape

            # Convert index coordinates to physical space
            Lz, Ly, Lx = self.domain
            x = np.arange(nx) * Lx / nx
            y = np.arange(ny) * Ly / ny
            z = np.arange(nz) * Lz / nz

            self._grid = pv.RectilinearGrid(x, y, z)

            # VTK is column‑major like Julia, so flatten in Fortran order
            self._grid["T"] = T.ravel(order="C")

        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=(not live), window_size=(800, 608))
            self._volume = self._plotter.add_volume(
                self._grid,
                scalars="T",
                cmap="turbo",
                clim=(cmin, cmax),
                opacity="linear",
            )
            self._plotter.add_axes()

            if live:
                self._plotter.show(auto_close=False, interactive_update=True)

        self._grid.point_data["T"][:] = T.ravel(order="C")
        if live:
            self._plotter.update(force_redraw=True)
            self._plotter.render()
        else:
            im = self._plotter.screenshot(return_img=True)
            self._plotter.close()
            self._plotter = None
            return im[:, :, :3]

    def close(self):
        if self.closed:
            return
        self.closed = True

        # Shut down the Julia simulation
        try:
            self.sim.shutdown_simulation()
            self.sim.GC.gc()
        except Exception as e:
            self.logger.warning(f"Could not shut down Julia simulation cleanly: {e}")

        # Shut down the PyVista plotter
        if self._plotter is not None:
            self._plotter.close()

        del self.sim
        super().close()
