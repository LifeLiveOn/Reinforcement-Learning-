import gymnasium as gym
import time
import argparse
import numpy as np
import torch
import cus_wrapper
import DQN
import collections

DEFAULT_ENV = "PongNoFrameskip-v4"
FPS = 25

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default=DEFAULT_ENV, help='Environment name')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA for training')
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to the trained model file")
    parser.add_argument("-r", "--record", help="Directory to save the video")
    parser.add_argument("--no-vis", default=True, dest='vis', action='store_false',
                        help="Disable rendering")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    env = cus_wrapper.make_env(args.env, "human" if args.vis else None)

    if args.record:
        env = gym.wrappers.RecordVideo(env, args.record)

    net = DQN.DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    state, _ = env.reset()
    total_reward = 0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.vis:
            env.render()

        state_v = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q_vals = net(state_v)
        action = torch.argmax(q_vals, dim=1).item()
        c[action] += 1

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if done:
            break

        if args.vis:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)

    if args.record:
        env.close()
