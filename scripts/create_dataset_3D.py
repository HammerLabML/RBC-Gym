import math
import rbc_gym  # noqa: F401
import os
import numpy as np
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
    steps = int(length // (dt * 4))  # dt is in freefall time units

    # dataset params
    dir = "data/datasets/3D"
    base_seed = 42

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
        # env params
        render_mode=None,
        rayleigh_number=ra,
        checkpoint=f"data/checkpoints/3D/{split}/ckpt_ra{ra}.h5",
        episode_length=length,
        state_shape=shape,
        heater_duration=dt,
        heater_segments=segments,
        heater_limit=limit,
        pressure=True,
    )

    # Set up h5 dataset
    path = f"{dir}/{split}/ra{ra}.h5"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as file:
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

        # states
        for idx in range(total_epsiodes):
            # states
            file.create_dataset(
                f"states{idx}",
                (steps, 6, shape[0], shape[1], shape[2]),
                chunks=(1, 6, shape[0], shape[1], shape[2]),
                compression="gzip",
                dtype=np.float32,
            )
            # actions
            file.create_dataset(
                f"actions{idx}",
                (steps, segments, segments),
                chunks=(steps, segments, segments),
                compression="gzip",
                dtype=np.float32,
            )
            # nusselts
            file.create_dataset(
                f"nusselts{idx}",
                (steps,),
                chunks=(steps,),
                compression="gzip",
                dtype=np.float32,
            )

    # Run environment and save observations
    batches = math.ceil(total_epsiodes / parallel_envs)
    for base_idx in tqdm(range(batches), desc="Total Episodes"):
        ids = [base_idx * parallel_envs + i for i in range(parallel_envs)]
        print(f"Processing episodes: {ids}")
        action = env.action_space.sample() * 0  # no control
        obs, info = env.reset(seed=[base_seed + id for id in ids])
        for step in tqdm(range(steps), position=1, desc="Time Steps", leave=False):
            # Save observations
            for idx, id in enumerate(ids):
                # only write if id is within the total episodes
                if id >= total_epsiodes:
                    continue
                # Save state, action, and nusselt number
                with h5py.File(path, "r+") as file:
                    file[f"states{id}"][step] = obs[idx]
                    file[f"actions{id}"][step] = action[idx]
                    file[f"nusselts{id}"][step] = info["nusselt"][idx]

            # Step environment
            obs, _, terminated, truncated, info = env.step(action)
            if truncated.any() or terminated.any():
                break

    env.close()


if __name__ == "__main__":
    # Set up multiprocessing
    mp.set_start_method("spawn", force=True)

    # Argument parser for command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset for RBC environment.")
    parser.add_argument(
        "--ra",
        type=int,
        default=2500,
        help="Rayleigh number for the simulation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split [train, val, test].",
    )
    ra = parser.parse_args().ra
    split = parser.parse_args().split

    # Create dataset
    print(f"Creating dataset for Rayleigh number: {ra}, split: {split}")
    if split == "train":
        create_dataset(ra=ra, split="train", total_epsiodes=20, parallel_envs=10)
    elif split == "test":
        create_dataset(ra=ra, split="test", total_epsiodes=10, parallel_envs=10)
    elif split == "val":
        create_dataset(ra=ra, split="val", total_epsiodes=5, parallel_envs=5)
    else:
        raise ValueError("Split must be one of [train, val, test].")
