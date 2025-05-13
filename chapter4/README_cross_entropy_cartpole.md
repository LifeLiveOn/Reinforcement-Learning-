# Cross Entropy Method for CartPole

## Overview

This implementation of the **Cross-Entropy Method** for solving the **CartPole-v1** problem uses reinforcement learning (RL) to map the observation space (state) of the environment to actions. The process involves using a neural network (NN) model to predict the best action based on the current state, with the goal of maximizing the total reward. The method leverages **cross-entropy loss** and **softmax** for action prediction.

### Procedure Overview

1. **Model Architecture**:

   - The model is a simple **feedforward neural network (NN)**. The input is the observation from the environment (CartPole) with 4 features (e.g., position, velocity, angle, angular velocity).
   - The network uses **hidden layers** to process the observation and outputs action probabilities (or logits). This output is then passed through a **softmax** layer (optional depending on your use of CrossEntropyLoss) to get the probability distribution of possible actions.
   - The **output layer** corresponds to the action space of the environment (in CartPole, two possible actions: move left or right).

2. **Cross-Entropy Loss**:

   - **Cross-Entropy Loss** is used to measure the difference between the predicted probabilities of the actions (as logits) and the actual actions taken by the agent in the environment. This loss is minimized to update the model's parameters.
   - **Softmax** is either applied in the model or handled by the `CrossEntropyLoss` function, depending on the approach.

3. **Iterating through Batches**:

   - During training, the algorithm processes batches of **observations** and **actions**.
   - For each batch, the model predicts the next action based on the current observation, executes it in the environment, and then collects the resulting **reward**.
   - The reward for each episode is accumulated in `total_reward`.

4. **Episode Steps**:

   - Each action the agent takes is recorded as an **EpisodeStep**, which stores the **observation** and **action** at that step.
   - Once an episode ends (i.e., when the agent reaches a terminal state or after a truncated episode), the step data is saved, and the batch is yielded for training.

5. **Next Observation (`next_obs`)**:

   - After each action, the agent receives a **next observation** (`next_obs`) from the environment. If the episode is terminated, the environment is reset, and the model starts a new episode with the updated state.

6. **Training and Evaluation**:
   - After processing the batch, the **model is updated** using the cross-entropy loss between predicted actions and the actual actions taken. The model is then trained iteratively with updated parameters until it achieves satisfactory performance (reward exceeds threshold).

## Cross-Entropy Method for CartPole: Detailed Steps

1. **Model Design**:

   - A neural network model is defined with the input layer size matching the observation space of the CartPole environment (4 features), one hidden layer, and an output layer representing the action space (2 actions).
   - **Softmax** or **CrossEntropyLoss** is used depending on whether you need the probability distribution or logits.

2. **Interaction with Environment**:

   - The agent selects actions based on predicted probabilities from the model.
   - The environment's response (`env.step(action)`) gives the **next observation**, **reward**, **done**, and other information.
   - The **reward** is accumulated to compute the **total_reward** for each episode.

3. **Recording the Step**:

   - Each step is recorded in a **named tuple** `EpisodeStep(observation, action)`, indicating the **observation** and **action** taken at each step of the episode.
   - This step data is used for training to update the model's policy.

4. **Resetting the Episode**:
   - If an episode terminates (`done=True`), the agent's state is reset (`next_obs = env.reset()`), and the batch is yielded for training.
   - After yielding, the model is updated using **cross-entropy loss**, and the environment starts a new episode with the updated state.

## Key Functions

### `NNmodel` (Model Definition)

- **Purpose**: Defines the neural network used to map observations to action probabilities.
- **Input**: Observation size (e.g., 4 features in CartPole).
- **Output**: Action probabilities (or logits).

### `iterate_batches` (Batch Collection)

- **Purpose**: Collects episodes from the environment and assembles them into batches for training.
- **Input**: Environment, model, batch size.
- **Output**: Yield a batch of episodes, including steps and rewards.

### `filter_batch` (Batch Filtering)

- **Purpose**: Filters out episodes based on their reward. Only episodes with rewards above a certain percentile are kept.
- **Input**: Batch of episodes, percentile.
- **Output**: Filtered observations and actions for training.

## Summary

This implementation uses the **Cross-Entropy Method** to train a neural network on the **CartPole-v1** environment. The agent selects actions based on a probability distribution predicted by the neural network, and the training process aims to maximize the total reward using **cross-entropy loss**.
