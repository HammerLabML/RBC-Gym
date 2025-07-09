import rbc_gym  # noqa: F401

from tqdm import tqdm
import gymnasium as gym

import multiprocessing as mp


def main() -> None:
    """Run a small vectorized rollout to sanity-check the environment."""
    # params
    dt = 0.5
    length = 50

    env = gym.make_vec(
        "rbc_gym/RayleighBenardConvection2D-v0",
        num_envs=5,
        vectorization_mode="async",
        vector_kwargs={
            "copy": True,
            "daemon": True,
        },
        render_mode="human",
        heater_duration=dt,
        episode_length=length,
    )

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    for _ in tqdm(range(int(length/dt))):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if truncated.any():
            break

    env.close()


if __name__ == "__main__":
    # On macOS and Windows the default “spawn” start‑method requires this guard.
    mp.set_start_method("spawn", force=True)
    main()
