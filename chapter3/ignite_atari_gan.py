import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

import torchvision.utils as vutils

import gymnasium as gym
from gymnasium import spaces
import ale_py
import numpy as np

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ignite_atari_gan.log')])
logger = logging.getLogger(__name__)

# Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 100

LEARNING_RATE = 0.0001
REPORT_INTERVAL = 100
SAVE_IMGS_INTERVAL = 1000

DIS_FILTER = 64
GEN_FILTER = 64

IMGS_SIZE = 64


class inputWrapper(gym.ObservationWrapper):
    """
    proprocess the input image to 64x64, and convert to float32, (C,H,W)
    """

    def __init__(self, *args):
        super(inputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, IMGS_SIZE, IMGS_SIZE),
            dtype=np.uint8)

    def observation(self, obs):
        obs = cv2.resize(obs, (IMGS_SIZE, IMGS_SIZE))

        obs = np.moveaxis(obs, -1, 0).astype(np.float32)
        return obs


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.pipeline = nn.Sequential(
            nn.Conv2d(input_shape[0], DIS_FILTER,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DIS_FILTER, DIS_FILTER * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DIS_FILTER * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DIS_FILTER * 2, DIS_FILTER * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DIS_FILTER * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DIS_FILTER * 4, DIS_FILTER * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DIS_FILTER * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DIS_FILTER * 8, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pipeline(x)
        return x.view(-1, 1).squeeze(1)  # (batch_size, 1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        self.pipeline = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, GEN_FILTER * 8,
                               kernel_size=4, stride=1, padding=0),  # stride = 1 means that the output size is same as input size, formula is (input_size - 1) * stride + kernel_size - 2 * padding
            nn.BatchNorm2d(GEN_FILTER * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(GEN_FILTER * 8, GEN_FILTER * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GEN_FILTER * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(GEN_FILTER * 4, GEN_FILTER * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GEN_FILTER * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(GEN_FILTER * 2, GEN_FILTER,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GEN_FILTER),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(GEN_FILTER, output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, x):
        return self.pipeline(x)  # (batch_size, channels, height, width)


def iterate_batches(envs, batch_size=BATCH_SIZE):
    """
    iterate the envs and yield a batch of images
    """
    # e.reset() fetch obs, info, so using [0] to get the obs
    batch = [e.reset()[0] for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)
    while True:
        # sample a random env
        e = next(env_gen)
        # sample a random action
        obs, reward, terminated, truncated, info = e.step(
            e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # normalize to [-1, 1]
            batch_np = np.stack(batch).astype(np.float32) / 127.5 - 1
            yield torch.tensor(batch_np)
            batch.clear()
        if terminated or truncated:
            # reset the env
            e.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atari GAN")
    parser.add_argument("--cuda", default=True,
                        action='store_true', help="Enable cuda computation")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Create the environment
    envs = [inputWrapper(gym.make(name, render_mode="rgb_array")) for name in [
        'ALE/Breakout-v5', 'ALE/Pong-v5', 'ALE/AirRaid-v5']]

    input_shape = envs[0].observation_space.shape

    discriminator = Discriminator(input_shape).to(device)
    generator = Generator(input_shape).to(device)

    # Create the optimizers
    objective = nn.BCELoss()
    optimizer_d = optim.Adam(discriminator.parameters(),
                             lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(),
                             lr=LEARNING_RATE, betas=(0.5, 0.999))

    true_labels = torch.ones(BATCH_SIZE).to(device)
    fake_labels = torch.zeros(BATCH_SIZE).to(device)

    def process_batch(trainer, batch):
        gen_input = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(device)
        gen_input.normal_(0, 1)

        real_images = batch.to(device)
        fake_images = generator(gen_input)

        # train discriminator
        optimizer_d.zero_grad()
        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images.detach())
        d_loss = objective(real_output, true_labels) + \
            objective(fake_output, fake_labels)

        d_loss.backward()
        optimizer_d.step()

        # train generator
        optimizer_g.zero_grad()
        fake_output = discriminator(fake_images)
        # how close the fake images to real images
        g_loss = objective(fake_output, true_labels)
        g_loss.backward()
        optimizer_g.step()

        if trainer.state.iteration % SAVE_IMGS_INTERVAL == 0:
            fake_images = vutils.make_grid(
                fake_images[:64], normalize=True, scale_each=True)
            trainer.tb.writer.add_image(
                "fake_images", fake_images, trainer.state.iteration)
            real_images = vutils.make_grid(
                real_images[:64], normalize=True, scale_each=True)
            trainer.tb.writer.add_image(
                "real_images", real_images, trainer.state.iteration)
            trainer.tb.writer.flush()

        return d_loss.item(), g_loss.item()  # 0 for d_loss, 1 for g_loss


# Create the engine
    # process_batch is the function that will be called for each batch
    # it should return a tuple of (d_loss, g_loss)
engine = Engine(process_batch)
tb = tb_logger.TensorboardLogger(log_dir=None)
engine.tb = tb

RunningAverage(output_transform=lambda x: x[1]).attach(engine, "avg_g_loss")
RunningAverage(output_transform=lambda x: x[0]).attach(engine, "avg_d_loss")

handler = tb_logger.OutputHandler(tag="training",
                                  metric_names=["avg_g_loss", "avg_d_loss"])
tb.attach(engine, log_handler=handler,
          event_name=Events.ITERATION_COMPLETED(every=REPORT_INTERVAL))

engine.run(data=iterate_batches(envs))
