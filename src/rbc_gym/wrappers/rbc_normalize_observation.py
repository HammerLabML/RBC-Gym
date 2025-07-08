from typing import Any

import gymnasium as gym
import numpy as np
from rbc_gym.envs import RayleighBenardConvection2DEnv, RayleighBenardConvection3DEnv


class RBCNormalizeObservation(gym.ObservationWrapper):
    """Normalize the observation to approximately lie in range [-1, 1]"""

    def __init__(
        self,
        env: RayleighBenardConvection2DEnv | RayleighBenardConvection3DEnv,
        heater_limit: float,
        maxval: int = 1,
        u_limit: int | None = 1.3,
        eps: float = 0.3,
        clip: bool = False,
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.heater_limit = heater_limit
        self.clip = clip
        self.maxval = maxval

        # fixed min and max values for temperature
        T = env.unwrapped.temperature_difference
        minT = T[0]
        maxT = T[1] + self.heater_limit

        # loose min and max for velocities
        if u_limit is None and isinstance(env.unwrapped, RayleighBenardConvection3DEnv):
            u_limit = self.__get_u_limit_3d(env.unwrapped.ra)
        elif u_limit is None:
            raise ValueError("u_limit must be provided for 2D RBC.")

        # reference values for each channel
        self.min_vals = [minT, -u_limit, -u_limit, -u_limit]
        self.max_vals = [maxT, u_limit, u_limit, u_limit]

        # new observation space
        shape = env.observation_space.shape
        limit = maxval + eps
        self.observation_space = gym.spaces.Box(
            low=-limit,
            high=limit,
            shape=shape,
            dtype=np.float32,
        )

    def observation(self, obs) -> Any:
        # Normalize each channel
        for c in range(obs.shape[0]):
            obs[c] = self.maxval * (
                2 * (obs[c] - self.min_vals[c]) / (self.max_vals[c] - self.min_vals[c])
                - 1
            )
        # at this point obs is in the range [-maxval, maxval], or outside of it if the observation is outside the min/max range
        if self.clip:
            obs = np.clip(obs, -self.maxval, self.maxval)

        return obs

    def __get_u_limit_3d(self, ra):
        w_inf = 0.96549382
        Ra_c = 654.37063331
        n = 1.06741877
        return w_inf * ra**n / (ra**n + Ra_c**n)
