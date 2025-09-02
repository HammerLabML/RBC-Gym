from gymnasium.envs.registration import register
import numpy as np

register(
    id="rbc_gym/RayleighBenardConvection2D-v0",
    entry_point="rbc_gym.envs:RayleighBenardConvection2DEnv",
    kwargs={
        "rayleigh_number": 10_000,
        "episode_length": 300,
        "observation_shape": (8, 48),
        "state_shape": (64, 96),
        "modes": 6,
        "actuator_limit": 0.75,
        "heater_duration": 1.5,
        "checkpoint": None,
        "use_gpu": False,
        "render_mode": None,
    },
)

register(
    id="rbc_gym/RayleighBenardConvection3D-v0",
    entry_point="rbc_gym.envs:RayleighBenardConvection3DEnv",
    kwargs={
        "rayleigh_number": 500,
        "prandtl_number": 0.7,
        "domain": [2, 2 * np.pi, 2 * np.pi],
        "state_shape": (32, 48, 48),
        "temperature_difference": [0, 1],
        "heater_segments": 8,
        "heater_limit": 0.9,
        "heater_duration": 0.125,
        "episode_length": 300,
        "checkpoint": None,
        "use_gpu": False,
        "render_mode": None,
    },
)
