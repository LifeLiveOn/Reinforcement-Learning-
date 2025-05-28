# RL using cross entropy for Cart Pole
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
BATCH_SIZE = 64
PERCENTILE = 70

# allow us to access the elements of the tuple by name
# instead of index
Episode = namedtuple('Episode', ['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', ['observation', 'action'])


class NNmodel(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size):
        super(NNmodel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
            # no softmax, because we will use CrossEntropyLoss (which has softmax and logits)
        )

    def forward(self, x):
        return self.net(x)  # (batch_size, action_size)


def iterate_batches(env, model, batch_size):
    """
    iterate over the environment and collect a batch of samples
    """
    batch = []
    episode_reward = 0.0
    episode_step = []
    obs = env.reset()[0]
    sm = nn.Softmax(dim=1)  # return the probability of each action

    while True:
        # each obs is a vector of four numbers in CartPole-v1
        # (Cartpole-v1) obs: [position, velocity, angle, angular_velocity]
        obs_v = torch.FloatTensor([obs])  # (1, 4)
        action_probs_v = sm(model(obs_v))  # (1, 2) [left, right]
        # because [0] return a numpy array of shape (1, action_size)
        # extract the first row with its columns
        act_probs = action_probs_v.data.numpy()[0]

        # p is the probability of each action [1, 2]
        # this is like the policy
        action = np.random.choice(len(act_probs), p=act_probs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        # curent obs and the action taken
        step = EpisodeStep(observation=obs, action=action)
        episode_step.append(step)

        if terminated or truncated:
            e = Episode(reward=episode_reward, steps=episode_step)
            batch.append(e)
            episode_reward = 0.0
            episode_step = []
            # seperate the next obs from the current obs to prevent the next obs from being added to the batch, did not use obs = env.reset() to allow us to use the last obs
            next_obs = env.reset()[0]
            if len(batch) == batch_size:
                yield batch
                batch.clear()
        obs = next_obs


def filter_batch(batch, percentile):
    """
    filter the episode in batch by the percentile
    """
    # batch is a list of episodes [Episode(reward, steps)]
    # example [100,200, 300] rewards for each episode in that batch
    rewards = list(map(lambda e: e.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = np.mean(rewards)

    train_obs = []
    train_act = []

    for reward, steps in batch:
        if reward < reward_bound:
            continue
        # instead of for i in range(len(steps)):
        train_obs.extend(map(lambda step: step.observation, steps))
        # train_obs.append(steps[i].observation)
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)  # (n_of_observation, obs_size)
    # (number of step, 1) #store integer64 because action is discrete
    # train_act = [0, 1, 0, 1, 0]  # List of actions taken by the agent (0 or 1)
    train_act_v = torch.LongTensor(train_act)

    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    ob_size = env.observation_space.shape[0]  # (1, 4)
    n_actions = env.action_space.n

    net = NNmodel(ob_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    writer = SummaryWriter('runs/cartpole')

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(
            batch, PERCENTILE)  # has batch size in each  variable
        # obs_v is a tensor of shape (n_of_observation, obs_size)
        # acts_v is a tensor of shape (n_of_action, 1)

        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print(
            f"iter: {iter_no}, loss: {loss_v.item()}, reward_bound: {reward_b}, reward_mean: {reward_m}")
        writer.add_scalar('loss', loss_v.item(), iter_no)
        writer.add_scalar('reward_bound', reward_b, iter_no)
        writer.add_scalar('reward_mean', reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()
