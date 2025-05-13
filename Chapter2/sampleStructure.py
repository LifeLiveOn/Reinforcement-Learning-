import random
class Environment:
    # This is a simple environment class that simulates an environment for an agent.
    # It has a fixed number of steps and returns observations and actions.
    # The agent interacts with the environment by taking actions and receiving observations.
    # The environment is done when the number of steps left reaches zero.
    # The observation is a list of three floats, and the action is a list of two integers.
    def __init__(self):
        self.steps_left = 10
    
    def get_observation(self) -> list[float]:
        return [0.0, 0.0, 0.0]
    
    def get_action(self) -> list[int]:
        return [0, 1]
    
    def is_done(self) -> bool:
        return self.steps_left == 0
    
    def action(self, action: int) -> float:
        # Simulate taking an action in the environment.
        # In a real environment, this would involve more complex logic.
        if self.is_done():
            raise Exception("Environment is done")
        self.steps_left -= 1
        return random.random() #return a random number from [0,1) as the reward
    

class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: Environment) -> None:
        # Get the current observation from the environment
        observation = env.get_observation()
        
        # Get the action from the environment
        actions = env.get_action()
        
        # Take an action in the environment and get the reward
        reward = env.action(random.choice(actions))
        
        # Update the total reward
        self.total_reward += reward
        
        # Check if the environment is done
        if env.is_done():
            print("Total Reward:", self.total_reward)
            return True
        return False


if __name__ == "__main__":
    env = Environment()
    agent = Agent()
    
    while not env.is_done():
        agent.step(env)
    print("Final Total Reward:", agent.total_reward)
    # The agent interacts with the environment until it is done.