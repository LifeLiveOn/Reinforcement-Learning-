import cv2
import gymnasium as gym

import gymnasium.spaces as spaces
import numpy as np
import collections
import ale_py


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip  # use _ to avoid conflict with gym's skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            observation, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            self._obs_buffer.append(observation)
            if term or trunc:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod  # this is a static method because it does not depend on the instance can be called without an instance
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3])
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3])
        else:
            raise ValueError(f"Unknown resolution: {frame.shape}")

        # Correctly apply grayscale to the reshaped image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Resize to 84x110 first (standard preprocessing step from DeepMind)
        resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)

        # Crop to 84x84 (focus on play area)
        cropped = resized[18:102, :]  # shape = (84, 84)

        # Add channel dimension
        x_t = np.reshape(cropped, (84, 84, 1)).astype(np.uint8)

        return x_t


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.n_steps = n_steps
        self.dtype = dtype

        # Assume input shape is (C, H, W)
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 3, "Expected observation shape (C, H, W)"
        c, h, w = obs_shape

        low = np.repeat(env.observation_space.low, n_steps, axis=0)
        high = np.repeat(env.observation_space.high, n_steps, axis=0)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=dtype
        )
        self.buffer = np.zeros((c * n_steps, h, w), dtype=dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buffer[...] = 0  # clear buffer
        for i in range(self.n_steps):
            self.buffer[i * obs.shape[0]:(i + 1) * obs.shape[0]] = obs
        return self.buffer, info

    def observation(self, observation):
        c = observation.shape[0]
        self.buffer[:-c] = self.buffer[c:]  # shift left
        self.buffer[-c:] = observation      # add new frame
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    """
    A wrapper that converts the observation from a numpy array to a PyTorch tensor.
    This is useful for environments where you want to use PyTorch for training.
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_space = self.observation_space.shape
        new_shape = (old_space[-1], old_space[0], old_space[1])
        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        """
        HWC (height, width,
    channel) to the CHW (channel, height, width) format required by PyTorch. The
    input shape of the tensor has a color channel as the last dimension, but Py
        """
        return np.moveaxis(observation, 2, 0)


class ScaleFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs, dtype=np.float32) / 255.0


def make_env(env_name, render_mode=None):
    """Create a Gym environment with the necessary wrappers applied.
    Returns:
        A wrapped Gym environment.
    """  # or any valid Gym environment name
    env = gym.make(env_name, render_mode=render_mode)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaleFloatFrame(env)
    return env
