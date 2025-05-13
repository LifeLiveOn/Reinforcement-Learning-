import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import torchvision.utils as vutils

import gymnasium as gym
from gymnasium import spaces
import ale_py #fix ale name space error
import numpy as np

import logging

# Configure global logging
logging.basicConfig(
    level=logging.INFO,  # Set log level here (e.g., DEBUG, INFO, WARNING)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),   # or "a" to append to the log file
    ]
)

log = logging.getLogger(__name__) # alias for the logger

LATENT_VECTOR_SIZE = 100 
#the random noise vector size for the generator -> generator received a random tensor of size (batch_size, LATENT_VECTOR_SIZE, 1, 1)
#the generator will output an image of size (batch_size, 3, 64, 64) thr transpose conv layers

DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 128
MAX_ITER = 100000
# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class InputWrapper(gym.ObservationWrapper):
    """
    Process of input numpy array:
    1. resize to IMAGE_SIZE x IMAGE_SIZE
    2. move color channel to the first dimension
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        #check if the observation space is a Box space
        assert isinstance(self.observation_space, spaces.Box)  
        #the observation space is a description of each observation
        #Box is a multi tensor, every element is in range [low, high]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, IMAGE_SIZE, IMAGE_SIZE), # every observation is a 3D tensor of the image (channels, height, width)
            dtype=np.float32
        )
        #when we obs.reset() , obs.shape = (3,64,64), obs.max() = 255, obs.min() = 0
        #if batch then we get (batch_size, 3, 64, 64)

    #the observation function is called every time the environment is reset,
    #observation is the image from the environment having shape (height, width, channels)
    def observation(self, observation):
        #resize the image
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        #move the color channel to the first dimension
        new_obs = np.moveaxis(new_obs, -1, 0)
        return new_obs.astype(np.float32)
    

class Discriminator(nn.Module):
    """
    Builds a Discriminator network for GANs.

    The discriminator takes an image as input and outputs a single value
    representing the probability that the input image is real (as opposed to generated).

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input image, e.g., (channels, width, height).

    Returns
    -------
    output : keras.Model or torch.nn.Module
        A model that outputs a single value between 0 and 1 for each input image.

    Example
    -------
    >>> model = discriminator((28, 28, 1))
    >>> output = model.predict(some_image)
    >>> print(output)  # Value between 0 and 1
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        #the pipline of the discriminator, for RGB images, the input shape is (3, 64, 64)
        #for grayscale images, the input shape is (1, 64, 64)
        #input shape is (channels, height, width)
        #output shape is (1, 1, 1)
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        conv_out = self.conv_pipe(x) # x is (batch_size, 1, 1, 1, 1)
        #flatten the output to (batch_size, 1)
        conv_out = conv_out.view(-1, 1).squeeze(dim=1) #ensure the output is 1D, reshape to (batch_size, 1) then squeeze to (batch_size,), -1 means automatically infer the batch size, 1 means the next dimension, squeeze to remove the dimension at index 1
        return conv_out
        

class Generator(nn.Module):
    """
    Builds a Generator network for GANs.
    The generator takes a random noise vector as input and outputs an image.

    Parameters:
    output_shape : tuple of int
        Shape of the output image, e.g., (channels, width, height).
    Returns:
    an image generated by the generator network.
    Example:
    >>> gen_input_v = torch.randn(BATCH_SIZE, 100, 1, 1) # random noise vector
    >>> model = generator(gen_input_v)

    """
    def __init__(self, output_shape):
        super(Generator,self).__init__()
        #the pipline of the generator, for RGB images, the output shape is (3, 64, 64)
        # for grayscale images, the output shape is (1, 64, 64)
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8, kernel_size=4, stride=1, padding=0), #padding 0 means no padding, stride 1 means the output size is the same as the input size
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4, kernel_size=4, stride=2, padding=1), #stride 2 means the output size is half of the input size, padding 1 means the output size is the same as the input size
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh() #output the image in the range of -1 to 1
        )        

    def forward(self, x):
        #x is the input to the generator, we call it by :
        #net_gener = Generator(output_shape=input_shape).to(device)
        #gen_output_v = net_gener(torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1))
        return self.pipe(x) #return the output of the generator


def iterate_batches(envs, batch_size):
    """
    Iterate over the environment and yield batches of images.
    """
    #e.reset() get (obs, info) and obs is the image from the environment
    #batch is just a list of images from the environment (env1, env2, env3....)
    batch = [e.reset()[0] for e in envs] #reset the environment and get the
    env_gen = iter(lambda: random.choice(envs), None) # get a random environment from the list of environments

    while iter_no < MAX_ITER:
        e = next(env_gen) #get the next environment
        obs, reward, terminated, truncated, info = e.step(e.action_space.sample()) #take a random action in the environment
        done = terminated or truncated
        if np.mean(obs) > 0.01: #check if the image is not black
            batch.append(obs) #append the observation to the batch
        if len(batch) == batch_size: #if 
            #normalize the batch to the range of -1 to 1
            #every batch has size ( 3, 64, 64)
            batch_np = np.stack(batch).astype(np.float32) * 2.0 / 255.0 - 1.0

            yield torch.tensor(batch_np,device=device) #return the full 1 batch as a tensor, load only the batch to the device instead of the whole batch
            batch.clear()
        if done:
            e.reset() #reset the environment if the episode is done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN on Atari games") 
    parser.add_argument("--cuda", default=True, action="store_true", help="Use GPU for training") #add an argument to use GPU for training

    args = parser.parse_args() #parse the arguments

    device = torch.device("cuda" if args.cuda else "cpu") #set the device to GPU if cuda is available, otherwise set it to CPU
    envs = [ #make 3 environments for the 3 games
        InputWrapper(gym.make(name, render_mode="rgb_array"))
        for name in ('ALE/Breakout-v5', 'ALE/Pong-v5', 'ALE/AirRaid-v5')
    ]

    input_shape =  envs[0].observation_space.shape # get the shape of the input image # (3, IMAGE_SIZE, IMAGE_SIZE) from our observation wrapper

    net_discr = Discriminator(input_shape).to(device) #create the discriminator and move it to the device # we only initialize the discriminator once
    net_gener = Generator(output_shape=input_shape).to(device) #create the generator and move it to the device initlaize the generator once
    #BCELoss(pred eg: [0.8], target is [1]) loss is ~ 0.2
    objectives = nn.BCELoss() #create the loss function
    #Binary Cross Entropy Loss, used for binary classification problems, take the output of the discriminator and the true labels (1 for real, 0 for fake) and calculate the loss

    #get the params of the generator and discriminator
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.7, 0.999)) #create the optimizer for the generator

    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.7, 0.999)) #create the optimizer for the discriminator
    writer = SummaryWriter(comment="atari_gan") #create the tensorboard writer

    gen_losses = [] #list to store the generator loss
    dis_losses = [] #list to store the discriminator loss

    iter_no = 0 #iteration number

    true_lavels_v = torch.ones(BATCH_SIZE, device=device) #create a tensor of ones for the true labels, has shape (BATCH_SIZE, 1)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device) #create a tensor of zeros for the fake labels

    best_avg_loss = float("inf")
    
    #we have 50000 iteration, we will get 50000 batches of images from the environments, each batch has size (BATCH_SIZE, 3, 64, 64)
    for batch_v in iterate_batches(envs, BATCH_SIZE):
        iter_no += 1 #increment the iteration number
        # fake samples, input is 4D tensor of shape (batch_size, channels, height, width)

        gen_input_v = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1, dtype=torch.float32, device=device) #create a random noise vector for the generator from a normal distribution

        batch_v = batch_v.to(device) #move the batch to the device
        gen_output_v = net_gener(gen_input_v) # Generator tạo ảnh giả

        #train the discriminator
        dis_optimizer.zero_grad()

        # Discriminator phân biệt ảnh thật và ảnh giả
        disoutput_true_v = net_discr(batch_v)                   # ảnh thật
        disoutput_fake_v = net_discr(gen_output_v.detach())     # ảnh giả (tách ra), ham detach() để không cập nhật weights của generator khi dang train discriminator

        dis_loss_v = objectives(disoutput_true_v, true_lavels_v) + objectives(disoutput_fake_v, fake_labels_v) #calculate the loss for the discriminator

        dis_loss_v.backward() #backpropagate the loss
        dis_optimizer.step() #update the weights of the discriminator
        dis_losses.append(dis_loss_v.item()) #append the loss to the list

        #train the generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v) #no DETACH TO TRAIN THE GENERATOR
        # Generator cố gắng làm cho Discriminator nghĩ ảnh giả là thật
        gen_loss_v = objectives(dis_output_v, true_lavels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        # Nếu số vòng lặp hiện tại chia hết cho REPORT_EVERY_ITER (ví dụ: mỗi 100 bước)
        if iter_no % REPORT_EVERY_ITER == 0:
            # In ra log thông tin: số vòng lặp, loss của generator và discriminator
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                    iter_no, np.mean(gen_losses),
                    np.mean(dis_losses))

            # Ghi loss của generator vào TensorBoard để quan sát đồ thị
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            
            # Ghi loss của discriminator vào TensorBoard
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)

            # Reset lại danh sách loss để ghi trung bình cho giai đoạn tiếp theo
            gen_losses = []
            dis_losses = []

        # Nếu số vòng lặp chia hết cho SAVE_IMAGE_EVERY_ITER (ví dụ: mỗi 1000 bước)
        # if iter_no == best_iteration:
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
                        # Ghi 64 ảnh giả đầu tiên sinh ra từ Generator vào TensorBoard
            writer.add_image("fake", vutils.make_grid(
                gen_output_v.data[:64], normalize=True), iter_no)

            # Ghi 64 ảnh thật đầu tiên trong batch vào TensorBoard để so sánh
            writer.add_image("real", vutils.make_grid(
                batch_v.data[:64], normalize=True), iter_no)
