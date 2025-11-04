import numpy as np
import rbc_gym  # noqa: F401
import gymnasium as gym
from tqdm import tqdm
from rbc_gym.utils.visualization import start_live_control, update_live_control

env = gym.make(
    "rbc_gym/RayleighBenardConvection2D-v0",
    render_mode="human",
    pressure=True,
    observation_shape=[64, 96],
    heater_duration=5,
)

plotter = start_live_control(
    modes=env.unwrapped.modes,
    heater_limit=env.unwrapped.heater_limit,
    Lx=2 * np.pi,  # matches Julia
    Δb=1.0,
    T0=2.0,
    show_modes=True,
    max_modes=6,  # or fewer if you want less clutter
)

obs, info = env.reset()
for step in tqdm(range(env.unwrapped.episode_steps)):
    action = env.action_space.sample()

    update_live_control(plotter, action)

    observation, reward, terminated, truncated, info = env.step(action)
    # print reward and nusselt number
    print(f"Step {step}: Reward = {reward:.3f}, Nu = {info['nusselt_state']:.3f}")
    env.render()
    if truncated:
        break
