import gym
from gym.wrappers import RecordVideo

# Create and wrap the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="recording", episode_trigger=lambda e: True)

total_reward = 0.0
total_steps = 0
obs, _ = env.reset()

# Random agent loop
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    total_steps += 1
    if terminated or truncated:
        break

print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
env.close()  # Flush video properly

#run using python .\APAININTHEAS.py