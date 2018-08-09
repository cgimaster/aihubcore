# import universe
import gym
import gym_pull

gym_pull.pull('github.com/ppaquette/gym-doom')        # Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('ppaquette/DoomBasic-v0')

observation_n = env.reset()

while True:
    env.render()
    action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
    observation_n, reward_n, done_n, info = env.step(action_n)
    print(observation_n)

