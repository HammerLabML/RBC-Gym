import rbc_gym  # noqa: F401
import vtk
import gymnasium as gym
from tqdm import tqdm


env = gym.make(
    "rbc_gym/RayleighBenardConvection3D-v0",
    render_mode="human",
    rayleigh_number=2500,
    heater_duration=0.25,
    episode_length=500,
    checkpoint="data/checkpoints/3D/train/ckpt_ra2500.h5",
)

obs, info = env.reset()
with tqdm(range(env.unwrapped.episode_length)) as pbar:
    while True:
        action = env.action_space.sample() * 0
        observation, reward, terminated, truncated, info = env.step(action)

        pbar.update(info["t"] - pbar.n)
        pbar.set_postfix({"reward": reward, "t": info["t"]})
        print("coming here, so completed the env.step() function")
        env.render()
        print("I completed the env.render() function")
        if truncated:
            break

env.close()
