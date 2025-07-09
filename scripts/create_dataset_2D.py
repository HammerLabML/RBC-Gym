import rbc_gym  # noqa: F401
import os
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import h5py
import multiprocessing as mp


def create_dataset(ra=10000, split="train", total_epsiodes=50, parallel_envs=5):
    # env params
    shape = (64, 96)
    dt = 0.5
    length = 500
    segments = 12
    limit = 0.75
    steps = int(length // dt)

    # dataset params
    dir = "data/datasets/2D"
    base_seed = 42

    # Set up environment
    env = gym.make_vec(
        "rbc_gym/RayleighBenardConvection2D-v0",
        num_envs=parallel_envs,
        vectorization_mode="async",
        vector_kwargs={
            "copy": True,
            "daemon": True,
        },
        render_mode=None,
        rayleigh_number=ra,
        episode_length=length,
        observation_shape=shape,
        heater_duration=dt,
        heater_segments=segments,
        heater_limit=limit,
        pressure=True,
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
        range(int(total_epsiodes / parallel_envs)), position=0, desc="Total Episodes", leave=True
    ):
        ids = [base_idx * parallel_envs + i for i in range(parallel_envs)]
        states = [
            file.create_dataset(
                f"states{id}",
                (steps, 5, shape[0], shape[1]),
                chunks=True,
                compression="gzip",
                dtype=np.float32,
            )
            for id in ids
        ]

        actions = [
            file.create_dataset(
                f"actions{id}",
                (steps, segments),
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
        for step in tqdm(range(steps), position=1, desc="Time Steps", leave=True, miniters=20):
            # Save observations
            for i, _ in enumerate(ids):
                states[i][step] = obs[i]
                actions[i][step] = action[i]
                nusselts[i][step] = info["nusselt_obs"][i]

            # Step environment
            obs, _, terminated, truncated, info = env.step(action)
            if truncated.any() or terminated.any():
                break

            print("test")

    env.close()
    file.close()


if __name__ == "__main__":
    # Set up multiprocessing
    mp.set_start_method("spawn", force=True)

    # Argument parser for command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Create dataset for RBC environment.")
    parser.add_argument(
        "--ra",
        type=int,
        default=10000,
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
        create_dataset(ra=ra, split="train", total_epsiodes=50, parallel_envs=20)
    elif split == "test":
        create_dataset(ra=ra, split="test", total_epsiodes=20, parallel_envs=20)
    elif split == "val":
        create_dataset(ra=ra, split="val", total_epsiodes=10, parallel_envs=10)
    else:
        raise ValueError("Split must be one of [train, val, test].")
