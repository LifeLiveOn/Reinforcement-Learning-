# ilustrate Q-Learning with OpenAI GYM frozen Lake
import gymnasium as gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake8x8-v1"
GAMMA = 0.9
TEST_EPISODES = 20  # Số lần chạy để kiểm tra chính sách hiện tại


class Agent:
    """
    Agent class to play the FrozenLake environment.
    Attributes:
        env: môi trường huấn luyện.
        state: The current state of the environment.
        rewards: lưu phần thưởng trung bình cho mỗi (state, action, next_state).
        visited: lưu số lần đã đi từ state qua action đến next_state.
        values: lưu giá trị ước lượng của mỗi trạng thái 𝑉(s).
    """

    def __init__(self):
        self.env = gym.make(ENV_NAME, desc=None, is_slippery=False)
        self.state = self.env.reset()[0]
        self.rewards = collections.defaultdict(float)
        # a dict of dicts ex: {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        self.visited = collections.defaultdict(collections.Counter)
        # a dict of floats ex: {0: 1.0, 1: 2.0}
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, n):
        """
        Choi n bước ngẫu nhiên trong môi trường de kham pha trạng thái.
        Args:
            n: số bước ngẫu nhiên để chơi.
        Ghi chú:
            reward from each  (state, action, next_state)
            count the number of times we have visited each state ex: {(0,0) : {0: 1, 1: 2}, (0,1): {0: 3, 1: 4}}

        Muc đich:
            Tinh Toan Q(s, a) 
        """
        for _ in range(n):
            action = self.env.action_space.sample()
            new_state, reward, truncated, terminated, info = self.env.step(
                action)
            # add reward ex: (0, 0, 1) += 1
            self.rewards[(self.state, action, new_state)] = reward
            self.visited[(self.state, action)][new_state] += 1
            if terminated or truncated:
                self.state = self.env.reset()[0]
            else:
                self.state = new_state

    def calc_action_value(self, state, action):
        """
        Calculate Q(s, a) for each action in state s. using bellman equation
        Q(s, a) = E[R(s, a, s')] + GAMMA * V(s')

        Args:
            state: the current state.
            action: the action to take.

        Example:
            we pass a tuple (state, action), 
            we going to get the counts visited for each state using this tuple
            ex: {(0, 0): {0: 1, 1: 2}, (0, 1): {0: 3, 1: 4}}
            total = sum(target_counts.values()) = 3 for (0, 0)

            for each target state, we calculate the reward and the value 

        Returns:
            the best action value for the state V(s) for the state s we passed
        """
        target_counts = self.visited[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)
                                  ]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    def select_action(self, state):
        """
        select the best action for the state s, we know we can get V(s) for each action because we need V(s) = Max(Q(s, a))
        this is the policy we are going to use to select the action
        Returns:
            the best action for the state s, s = argmax(Q(s, a))
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        """
        we simulate an episode to get the reward, using our current policy
        Ghi nhận phần thưởng mới để cải thiện giá trị Q(s, a)
        Args:
            env: the environment to play in.
        Returns:
            the total reward for the episode.
        Ghi chú:
         test bang cach de agent chon best action tu state 0 , di toi uu tu do de ktra xem reward minh dat duoc co tot khong, dang ktra cai policy minh da chon
        """
        total_reward = 0.0
        state = env.reset()[0]
        done = False
        while True:
            action = self.select_action(state)
            new_state, reward, truncated, terminated, info = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.visited[(state, action)][new_state] += 1
            total_reward += reward
            if terminated or truncated:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        """
        for each state, we calculate the best V(s) for each s
        then we get v(s) = max(Q(s, a)) for each state s
        Returns
        the best V(s) for each state s
        """
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    """
    We use the value iteration algorithm to solve the FrozenLake environment.
    we create 2 environments:
    1. the training environment: used to collect training data and random exploration.
    2. the test environment: used to evaluate the current policy (greedy action selection).
    The test environment is NOT used to collect training data or random exploration.

    update through the value iteration algorithm
    evaluate the current policy using the test environment 
    """
    test_env = gym.make(ENV_NAME, desc=None, is_slippery=False)
    test_env = gym.make(ENV_NAME, desc=None, is_slippery=False)

    agent = Agent()
    writer = SummaryWriter("runs/frozenlake-value-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1

        agent.play_n_random_steps(100)  # poulate the rewards and visited dicts
        # calculate the best V(s) for each state s, also update the values dict
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            # simulate an episode to get the reward, also update the rewards and visited dicts
            reward += agent.play_episode(test_env)
        # the reward is the sum average of all rewards in the episode
        reward /= TEST_EPISODES
        print(f"Iter: {iter_no}, reward: {reward}")
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
