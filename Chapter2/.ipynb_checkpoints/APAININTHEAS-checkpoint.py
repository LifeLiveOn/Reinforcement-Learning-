#CartPole session
import gym
e = gym.make('CartPole-v1')

obs = e.reset()
print(obs) # [ 0.00000000e+00  0.00000000e+00  2.40000000e+00 -1.20000000e-01], we have 4 observations data
print(e.action_space) # Discrete(2) , 2 actions move 0 or 1 0 to left, 1 to right
print(e.observation_space) # Box(4,) number of observations

print(e.step(0)) # take action 0 go to left, we will get a tuple of 4 values new observation
#(array([-0.02894881, -0.2209735 ,  0.01005542,  0.33839016]), 1.0, False, {})
#reward is 1.0, done is False, and info is empty

print(e.action_space.sample()) # take a random action
print(e.observation_space.sample()) # take a random observation