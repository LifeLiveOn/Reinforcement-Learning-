import cus_wrapper
import DQN

import argparse
import time
import numpy as np

import collections
import torch
import torch.nn as nn

import torch.optim as optim

from tensorboardX import SummaryWriter

import logging

# Basic config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("chapter6/training.log"),  # writes to file
        logging.StreamHandler()               # prints to console
    ]
)

DEFAULT_ENV = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0

GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_SiZE = 10000  # kích thước của bộ nhớ bufer để lưu trữ các trải nghiệm
# số bước ngẫu nhiên để chơi trước khi bắt đầu huấn luyện, skip cac frames intro
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
# số bước để đồng bộ hóa trọng số của mạng chính và mạng mục tiêu
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000  # số bước để giảm epsilon từ 1.0 đến 0.01
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

Experience = collections.namedtuple('Experience',
                                    ['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceBuffer:
    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # take a sample of batches from the buffer
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()[0]
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device='cuda'):
        done_reward = None
        # in the first iterations we will use random actions
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # self.state is a numpy array
            state_action = torch.from_numpy(
                np.array([self.state], dtype=np.float32)).to(device)
            state_values = torch.tensor(state_action).to(device)
            q_vals_v = net(state_values)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward

        exp = Experience(
            state=self.state,
            action=action,
            reward=reward,
            done=done,
            next_state=new_state
        )
        self.exp_buffer.append(exp)
        self.state = new_state
        if done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, target_net, device='cuda'):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.from_numpy(states).float().to(device)
    next_states_v = torch.from_numpy(next_states).float().to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0  # if done, next state value is 0
    next_state_values = next_state_values.detach()  # detach from the graph
    expected_state_action_values = rewards_v + GAMMA * next_state_values
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default=DEFAULT_ENV, help='Environment name')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA for training')
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    env = cus_wrapper.make_env(args.env)
    print("Observation shape:", env.observation_space.shape)
    net = DQN.DQN(env.observation_space.shape, env.action_space.n).to(device)

    target_net = DQN.DQN(env.observation_space.shape,
                         env.action_space.n).to(device)

    writer = SummaryWriter('chapter6/runs/' + args.env + '-' +
                           time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    # print(net)

    buffer = ExperienceBuffer(REPLAY_SiZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)
        rewrad = agent.play_step(net, epsilon, device=device)
        if rewrad is not None:
            total_rewards.append(rewrad)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            logging.info("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" %
                         (frame_idx, len(total_rewards), mean_reward, epsilon, speed))

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward/mean", mean_reward, frame_idx)
            writer.add_scalar("reward/total", rewrad, frame_idx)

            if best_mean_reward is None or mean_reward > best_mean_reward:
                # torch.save(net.state_dict(), 'best_model.pth')
                print("Best model saved")

                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f" %
                          (best_mean_reward, mean_reward))
                else:
                    print("Best mean reward initialized at %.3f" % mean_reward)

                best_mean_reward = mean_reward

            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                torch.save(net.state_dict(), 'chapter6/solved_model.pth')
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())
            print("Target net synced")

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, target_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
