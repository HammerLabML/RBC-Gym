import rbc_gym  # noqa: F401
import gymnasium as gym
from tqdm import tqdm

env = gym.make(
    "rbc_gym/RayleighBenardConvection2D-v0",
    render_mode="human",
    pressure=True,
    observation_shape=[64, 96],
)

obs, info = env.reset()
for step in tqdm(range(env.unwrapped.episode_steps)):
    action = env.action_space.sample() * 0
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if truncated:
        break
