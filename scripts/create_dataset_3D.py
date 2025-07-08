import rbc_gym  # noqa: F401
import os
import numpy as np
from rbc_gym.wrappers import RBCNormalizeObservation
import gymnasium as gym
from tqdm import tqdm
import h5py
import multiprocessing as mp


def create_dataset(ra=2500, split="train", total_epsiodes=1, parallel_envs=1):
    # env params
    shape = (16, 32, 32)
    dt = 0.375
    length = 300
    segments = 8
    limit = 0.9
    steps = int(length // (dt*4)) # dt is in freefall time units

    # dataset params
    dir = "data/datasets/3D"
    base_seed = 42

    def wrap(env):
        return RBCNormalizeObservation(env, heater_limit=limit, u_limit=None)

    # Set up environment
    env = gym.make_vec(
        "rbc_gym/RayleighBenardConvection3D-v0",
        # vectorization params
        num_envs=parallel_envs,
        vectorization_mode="async",
        vector_kwargs={
            "copy": True,
            "daemon": True,
        },
        # wrappers
        wrappers=[wrap],
        # env params
        render_mode="human",
        rayleigh_number=ra,
        checkpoint=f"data/checkpoints/3D/{split}/ckpt_ra{ra}.h5",
        episode_length=length,
        state_shape=shape,
        heater_duration=dt,
        heater_segments=segments,
        heater_limit=limit,
    )

    # Set up h5 dataset
    path = f"{dir}/{split}/ra{ra}.h5"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file = h5py.File(path, "w")

    # Save commonly used parameters of the simulation
    file.attrs["episodes"] = total_epsiodes
    file.attrs["steps"] = steps
    file.attrs["ra"] = ra
    file.attrs["shape"] = shape
    file.attrs["dt"] = dt
    file.attrs["timesteps"] = length
    file.attrs["segments"] = segments
    file.attrs["limit"] = limit
    file.attrs["base_seed"] = base_seed

    # Run environment and save observations
    for base_idx in tqdm(
        range(int(total_epsiodes / parallel_envs)), position=0, desc="Total"
    ):
        ids = [base_idx * parallel_envs + i for i in range(parallel_envs)]
        states = [
            file.create_dataset(
                f"states{id}",
                (steps, 4, shape[0], shape[1], shape[2]),
                chunks=True,
                compression="gzip",
                dtype=np.float32,
            )
            for id in ids
        ]

        actions = [
            file.create_dataset(
                f"actions{id}",
                (steps, segments, segments),
                chunks=True,
                compression="gzip",
                dtype=np.float32,
            )
            for id in ids
        ]

        nusselts = [
            file.create_dataset(
                f"nusselts{id}",
                (steps,),
                chunks=True,
                compression="gzip",
                dtype=np.float32,
            )
            for id in ids
        ]

        action = env.action_space.sample() * 0  # no control
        obs, info = env.reset(seed=[base_seed + id for id in ids])
        for step in tqdm(range(steps), position=1, desc="Running", leave=False):
            # Save observations
            for i, _ in enumerate(ids):
                states[i][step] = obs[i]
                actions[i][step] = action[i]
                nusselts[i][step] = info["nusselt"][i]

            # Step environment
            obs, _, terminated, truncated, info = env.step(action)
            env.render()
            if truncated.any() or terminated.any():
                break

    env.close()
    file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset for RBC environment.")
    parser.add_argument(
        "--ra",
        type=int,
        default=2500,
        help="Rayleigh number for the simulation.",
    )

    mp.set_start_method("spawn", force=True)
    ra = parser.parse_args().ra
    create_dataset(ra=ra, split="train", total_epsiodes=30, parallel_envs=10)
    create_dataset(ra=ra, split="test", total_epsiodes=10, parallel_envs=10)
    create_dataset(ra=ra, split="val", total_epsiodes=5, parallel_envs=5)
