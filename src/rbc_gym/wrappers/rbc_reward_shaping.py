import logging
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from rbc_gym.envs.rbc2D import RBCField
from typing import Any, Dict, Optional, Tuple


class RBCRewardShaping(gym.Wrapper):
    """
    Wrapper to shape the reward based on the distance of Benard cells.
    """

    def __init__(
        self, env: gym.Env, shaping_weight: float, debug_cell_dist: bool = False
    ):
        super().__init__(env)

        self.logger = logging.getLogger(__name__)
        self.shaping_weight = shaping_weight
        self.debug_cell_dist = debug_cell_dist
        self.size_state = env.unwrapped.state_shape

        if debug_cell_dist:
            # NOTE Just for debugging the cell distance computation
            self.fig_anim, self.ax_anim = plt.subplots()
            self.ax_anim.set_xlim(0, 2 * np.pi)
            self.ax_anim.set_ylim(-2, 2)
            (self.line,) = self.ax_anim.plot(
                np.linspace(0, 2 * np.pi, 96), np.linspace(1, 2, 96), "b-"
            )  # just some initial values for plotting
            (self.line_uy,) = self.ax_anim.plot(
                np.linspace(0, 2 * np.pi, 96), np.linspace(1, 2, 96), "r-"
            )
            (self.line_TuY,) = self.ax_anim.plot(
                np.linspace(0, 2 * np.pi, 96), np.linspace(1, 2, 96), "g-"
            )
            (self.line_cells,) = self.ax_anim.plot([], [], "x")

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Any, Dict[str, Any]]:
        # NOTE: for debugging the cell distance computation
        if self.debug_cell_dist:
            self.update()
            plt.show(block=False)

        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        # NOTE For debugging, plot mid-line temperature and velocity.
        if self.debug_cell_dist:
            self.update()

        # call original step
        obs, reward, closed, truncated, info = self.env.step(action)

        # compute the distance between the Bénard cells
        state = info["state"]
        cd = self.compute_cell_distances(state)
        reward = self.__apply_reward_shaping(cd, reward)

        # add to info dict
        info["cell_dist"] = cd

        return obs, reward, closed, truncated, info

    def __apply_reward_shaping(self, cell_distances, reward) -> float:
        w = self.shaping_weight
        # reward shaping
        # NOTE: works for our specific horizontal domain, needs
        # simple modification to generalize
        # scale to [0, 1], 0 is close, 1 is far (maximum distance is pi)
        cd_normalized = (-cell_distances + np.pi) / np.pi
        reward = (1 - w) * reward + w * cd_normalized

        if np.isnan(reward):
            self.logger.error("Reward is NaN")

        return reward

    def compute_cell_distances(self, state, use_avg=False) -> float:
        """
        Computes the distance between the Bénard cells of the state, given the mid-line temperature
        NOTE: works for our specific horizontal domain, needs simple modification to generalize
        use_avg is a boolean that determines whether the column average of the y-velocity field is used as a signal to find the cells
        """
        distance = 0
        distances = []
        if use_avg:
            uy = state[RBCField.UY].mean(axis=0)
        else:
            uy = state[RBCField.UY][int(self.size_state[0] / 2) - 1]

        # find peaks in the y-velocity field
        peaks, _ = find_peaks(uy, height=0.001)

        # pick out the two largest peaks
        # in addition: one can add a check of finding peaks in the y-velocity field, the cell locations are always at the maxima of the y-velocity
        # for example, only consider peaks where the y-velocity is positive
        # peaks_candidates = peaks_candidates[uy[peaks_candidates] > 0]

        domain_x = np.linspace(
            0, 2 * np.pi, self.size_state[1], endpoint=False
        )  # periodic domain
        if len(peaks) <= 1:
            distance = 0  # only one peak, no distance, it's the optimal situation.
        elif len(peaks) >= 2:
            # Compute distance between all combinations of peaks
            nr_pairs = int(len(peaks) * (len(peaks) - 1) / 2)
            distances = np.zeros(nr_pairs)
            k = 0
            for i in range(len(peaks)):
                for j in range(i + 1, len(peaks)):
                    dist1 = np.abs(domain_x[peaks[j]] - domain_x[peaks[i]])
                    dist2 = 2 * np.pi - dist1
                    distances[k] = min(dist1, dist2)
                    # Check if there is no negative velocity between the cells, because if there is, we consider them to be the same cell.
                    if dist1 < dist2:
                        if np.all(uy[peaks[i] : peaks[j]] > 0):
                            distances[k] = 0
                    else:
                        if np.all(uy[peaks[j] :] > 0) and np.all(uy[: peaks[i]] > 0):
                            distances[k] = 0
                    k += 1
            # NOTE for mean distance, use the line below
            # distance = np.sum(distances) / nr_pairs
            # NOTE for maximum distance, use the line below
            distance = np.max(distances)

        if self.debug_cell_dist:
            self.line_cells.set_data(domain_x[peaks], uy[peaks])
            print(
                f"Distance between cells: {distance}. Number of peaks: {len(peaks)}, max distance: {distance}"
            )

        return distance

    def update(self):
        """NOTE Michiel: I wrote this function for debugging the cell distance computation. It plots the mid-line temperature and velocity."""
        state = self.get_state()
        T_mid_line = state[RBCField.T][int(self.size_state[0] / 2) - 1]
        uy = state[RBCField.UY][int(self.size_state[0] / 2) - 1]
        xdata = np.linspace(0, 2 * np.pi, self.size_state[1], endpoint=False)
        ydata = T_mid_line

        self.line.set_data(xdata, ydata)
        self.line_uy.set_data(xdata, uy)
        self.line_TuY.set_data(xdata, (T_mid_line - 1.5) * uy)

        self.fig_anim.canvas.draw()
        self.fig_anim.canvas.flush_events()
