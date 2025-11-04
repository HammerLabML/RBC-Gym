import logging
import warnings
import os
from pathlib import Path
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import matplotlib

import gymnasium as gym
import pygame
import juliacall
import juliapkg
import numpy as np


class RBCField(IntEnum):
    T = 0
    UX = 1
    UY = 2
    P = 3


def colormap(value, vmin=1, vmax=2, colormap="turbo"):
    cmap = matplotlib.colormaps[colormap]
    value = (value - vmin) / (vmax - vmin)
    return cmap(value, bytes=True)[:, :, :3]


class RayleighBenardConvection2DEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        rayleigh_number: Optional[int] = 10_000,
        episode_length: Optional[int] = 300,
        observation_shape: Optional[list] = [8, 48],
        state_shape: Optional[list] = [64, 96],
        modes: Optional[int] = 6,
        coeff_scheme: Optional[str] = "equal",
        heater_limit: Optional[float] = 0.75,
        heater_duration: Optional[float] = 1.5,
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
        self.episode_length = episode_length
        self.observation_shape = observation_shape
        self.state_shape = state_shape
        self.temperature_difference = [1, 2]
        self.modes = modes  # number of Fourier mode pairs (cos/sin)
        self.heater_limit = (
            heater_limit  # fraction of Δb allowed for bottom fluctuation
        )
        self.heater_duration = heater_duration
        self.include_pressure = pressure
        self.episode_steps = int(episode_length / heater_duration)

        # Coefficient scaling: map NN outputs in [-1, 1] to physical coeffs
        self.coeff_scheme = coeff_scheme
        self.per_coeff_max = self._compute_coeff_bounds(
            self.modes, self.heater_limit, scheme=self.coeff_scheme
        )

        # Print environment configuration
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using Rayleigh number Ra={self.ra}")
        self.logger.info(f"Using episode length {self.episode_length} timesteps")

        # The agent takes actions between [-1, 1] on the Fourier coefficients
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(2 * self.modes,), dtype=np.float32
        )

        # Observation Space
        channels = 3
        lows = [
            np.ones(self.observation_shape) * 1,
            np.ones(self.observation_shape) * (-np.inf),
            np.ones(self.observation_shape) * (-np.inf),
        ]
        highs = [
            np.ones(self.observation_shape) * 2 + self.heater_limit,
            np.ones(self.observation_shape) * np.inf,
            np.ones(self.observation_shape) * np.inf,
        ]

        if self.include_pressure:
            channels += 2
            lows.append(np.ones(self.observation_shape) * (-np.inf))
            lows.append(np.ones(self.observation_shape) * (-np.inf))
            highs.append(np.ones(self.observation_shape) * np.inf)
            highs.append(np.ones(self.observation_shape) * np.inf)

        self.observation_space = gym.spaces.Box(
            np.stack(lows, dtype=np.float32, axis=0),
            np.stack(highs, dtype=np.float32, axis=0),
            shape=(
                channels,
                self.observation_shape[0],
                self.observation_shape[1],
            ),
            dtype=np.float32,
        )

        # Julia simulation
        juliapkg.resolve()
        self.sim = juliacall.newmodule("RBCGymAPI")
        package_dir = os.path.dirname(os.path.abspath(__file__))
        julia_file_path = os.path.join(package_dir, "..", "sim", "rbc_sim2D_api.jl")
        self.sim.include(julia_file_path)

        # Rendering
        self.render_mode = render_mode
        self.screen_width = 768
        self.screen_height = 512
        self.screen = None
        self.clock = None

    def _compute_coeff_bounds(
        self, modes: int, actuator_limit: float, Δb: float = 1.0, scheme: str = "decay"
    ) -> np.ndarray:
        """
        Per-coefficient maximum magnitudes so that the Julia-side scaling K≈1 most of the time.
        Returns an array of shape (2*modes,) with bounds for [a1,b1,a2,b2,...].

        scheme="equal": same bound for all coefficients (L1 budget split equally)
        scheme="decay": give low-k more room via 1/n decay (recommended)
        """
        if scheme == "equal":
            m = actuator_limit * Δb / (2 * modes)
            return np.full(2 * modes, m, dtype=np.float32)
        # decay ~ 1/n
        HN = sum(1.0 / k for k in range(1, modes + 1))
        bounds = []
        for n in range(1, modes + 1):
            w_pair = actuator_limit * Δb * (1.0 / n) / HN  # pair budget
            bounds += [w_pair / 2, w_pair / 2]  # split equally to a_n and b_n
        return np.array(bounds, dtype=np.float32)

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
            sensors=self.observation_shape[::-1],  # julia uses column-major order
            grid=self.state_shape[::-1],  # julia uses column-major order
            modes=self.modes,
            actuator_limit=self.heater_limit,
            dt=self.heater_duration,
            seed=self.np_random_seed,
            checkpoint_path=path,
            use_gpu=self.use_gpu,
        )

        # Reset action
        self.last_action = self.action_space.sample() * 0
        self.last_coeffs = np.zeros_like(self.per_coeff_max, dtype=np.float32)

        return self.__get_obs(), self.__get_info()

    def step(self, action: Any = None) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        terminated = False  # is always false; no terminal state
        truncated = False
        # zero action if none
        if action is None:
            action = np.zeros(self.action_space.shape, dtype=np.float32)
            warnings.warn("No action provided, using zero action")

        # Scale NN action in [-1,1] to physical Fourier coefficients
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        coeffs = action * self.per_coeff_max  # elementwise
        self.last_coeffs = coeffs.copy()

        # Simulation Step
        success = self.sim.step_simulation(np.array(coeffs))
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

    def __get_state(self) -> Any:
        state = np.array(self.sim.get_state(), dtype=np.float32)
        if not self.include_pressure:
            state = state[:3, :, :]
        state = state.transpose(0, 2, 1)  # julia uses column-major order
        return state

    def __get_obs(self) -> Any:
        obs = np.array(self.sim.get_observation(), dtype=np.float32)
        if not self.include_pressure:
            obs = obs[:3, :, :]
        obs = obs.transpose(0, 2, 1)  # julia uses column-major order
        return obs

    def __get_reward(self) -> float:
        nu = self.sim.get_nusselt(state=False)
        return -nu

    def __get_info(self) -> dict[str, Any]:
        t, step = self.sim.get_info()
        nu_state = self.sim.get_nusselt(state=True)
        nu_obs = self.sim.get_nusselt(state=False)
        sumabs = float(np.abs(self.last_coeffs).sum())
        K_est = max(1.0, sumabs / (self.heater_limit))
        return {
            "t": t,
            "step": step,
            "nusselt_state": nu_state,
            "nusselt_obs": nu_obs,
            "state": self.__get_state(),
            "coeff_sumabs": sumabs,
            "K_est": K_est,
        }

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
            )
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Rayleigh Benard Convection")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Data
        data = self.__get_state()[RBCField.T]
        data = np.transpose(data)
        data = np.flip(data, axis=1)  # orgin in pygame is top left
        data = colormap(data, vmin=1, vmax=2 + self.heater_limit)

        if self.render_mode == "human":
            canvas = pygame.Surface((96, 64))
            pygame.surfarray.blit_array(canvas, data)

            # scale canvas
            canvas = pygame.transform.scale(
                canvas, (self.screen_width, self.screen_height)
            )
            self.screen.blit(canvas, (0, 0))

            # Show screen
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None

        elif self.render_mode == "rgb_array":
            return data.transpose(1, 0, 2)

        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
