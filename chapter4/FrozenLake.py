# use cross entropy grid world category , size 4x4
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/
import gymnasium as gym
import ale_py
import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import utils as vutils

import numpy as np

import logging
from collections import namedtuple


HIDDEN_SIZE = 128  # neuronos in the hidden layer
BATCH_SIZE = 100
PERCENTILE = 70
GAMMA = 0.9  # discount factor

# our environment is 4x4 grid world, observation is just a number from 0 to 15
# 0 mean at (0,0) and 15 mean at (3,3), action space í left, dơn, right , up (0, 1, 2, 3)

# idea is to have a neural network that takes the observation as input and outputs the action, like a state machine the input has len 16, 1 for the current state and 0 for the rest


class InputWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(InputWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n, )  # (16, )
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float32
        )

    def observation(self, observation):
        # create a one hot vector of size 16
        res = np.copy(self.observation_space.low)  # an array of 16 zeros
        # if observation == 5 then [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        res[observation] = 1
        return res


class NnModule(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size):
        super(NnModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
            # no softmax, because we will use CrossEntropyLoss (which has softmax and logits)
        )

    def forward(self, x):
        return self.net(x)  # (batch_size, action_size)


Episode = namedtuple('Episode', ['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', ['observation', 'action'])


def iterate_batches(env, model, batch_size):
    batch = []
    episode_reward = 0.0
    episode_step = []
    obs = env.reset()[0]  # get the first observation
    sm = nn.Softmax(dim=1)  # return the probability of each action

    # iterate over the environment and collect a batch of samples
    while True:
        obs_v = torch.FloatTensor(obs).unsqueeze(0).to(device)  # (1, 16)
        action_probs_v = sm(model(obs_v))  # (1, 4) [left, down, right, up]
        action_probs = action_probs_v.cpu().detach().numpy()[
            0]  # (4, ) #first row
        action = np.random.choice(len(action_probs), p=action_probs)
        next_obs, reward, truncated, terminated, info = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_step.append(step)

        if terminated or truncated:
            batch.append(Episode(reward=episode_reward, steps=episode_step))
            episode_reward = 0.0
            next_obs = env.reset()[0]
            episode_step = []
            if len(batch) == batch_size:
                yield batch
                batch.clear()
        obs = next_obs


# def filter_batch(batch, percentile):
#     rewards = [e.reward for e in batch]
#     reward_bound = np.percentile(rewards, percentile)
#     reward_mean = np.mean(rewards)

#     train_obs = []
#     train_act = []

#     for episode in batch:
#         if episode.reward < reward_bound:
#             continue
#         train_obs.extend([step.observation for step in episode.steps])
#         train_act.extend([step.action for step in episode.steps])
#     train_obs_v = torch.FloatTensor(train_obs).to(device)
#     train_act_v = torch.LongTensor(train_act).to(device)
#     return train_obs_v, train_act_v, reward_bound, reward_mean

def filter_batch(batch, percentile):
    # the less the step the better reward we get
    def filter_fun(s): return s.reward * (GAMMA ** len(s.steps))
    disc_rewards = list(map(filter_fun, batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation,
                                 example.steps))
            train_act.extend(map(lambda step: step.action,
                                 example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = InputWrapper(gym.make('FrozenLake-v1', render_mode='human',
                       desc=None, map_name="4x4", is_slippery=False))
    obs_size = env.observation_space.shape[0]  # 16
    action_size = env.action_space.n  # 4
    model = NnModule(obs_size, HIDDEN_SIZE, action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter('runs/frozenlake-naive')
    objective = nn.CrossEntropyLoss()

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(
            env, model, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(
            lambda s: s.reward, batch))))

        # Kết hợp dữ liệu: full_batch + batch kết hợp tất cả các episode thu thập được từ nhiều batch.
        full_batch, obs, acts, reward_bound = \
            filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue
        obs_v = torch.FloatTensor(obs).to(device)
        acts_v = torch.LongTensor(acts).to(device)
        full_batch = full_batch[-500:]  # keep the last 500 episodes

        optimizer.zero_grad()
        action_scores_v = model(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, rw_mean=%.3f, "
              "rw_bound=%.3f, batch=%d" % (
                  iter_no, loss_v.item(), reward_mean,
                  reward_bound, len(full_batch)))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        if reward_mean > 0.8:
            print("Solved!")
            break
    writer.close()
