# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
import gym
# Wiki:
# https://github.com/openai/gym/wiki/CartPole-v0
# Environment page:
# https://gym.openai.com/envs/CartPole-v0

# get the environment
env = gym.make('CartPole-v0')

# put yourself in the start state
# it also returns the state
#initial state: cart position, cart velocity, pole angle, pole velocity at tip
initial_state = env.reset()
# Out[50]: array([-0.04533731, -0.03231478, -0.01469216,  0.04151   ])

# what do the state variables mean?
# Num Observation Min Max
# 0 Cart Position -2.4  2.4
# 1 Cart Velocity -Inf  Inf
# 2 Pole Angle  ~ -41.8°  ~ 41.8°
# 3 Pole Velocity At Tip  -Inf  Inf
print(initial_state)

#get state space
box = env.observation_space

print(box)
print(box.contains)

N = 100
total_reward = 0
for i in range(N):
    done = False
    while not done:
        observation, reward, done, info = env.step(action=env.action_space.sample())
        #print(observation, reward)
        total_reward = total_reward + 1
    env.reset()
print('Random algorithm has avg total reward: {} ',total_reward/N)