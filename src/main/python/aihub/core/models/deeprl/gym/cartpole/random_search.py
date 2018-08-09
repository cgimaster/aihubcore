import gym
import numpy as np
from aihub.core.common import settings
from gym import wrappers

env = gym.make('CartPole-v0')
N = 20
M = 10
best_weights = np.random.rand(4,1)
best_result = 0
np.random.seed(1)
for i in range(N):
    new_weights = np.random.rand(4,1)
    total_reward = 0
    for m in range(M):
        done = False
        state = env.reset()
        while not done:
            action = 1 if state.dot(new_weights) > 0 else 0
            observation, reward, done, info = env.step(action=action)
            total_reward = total_reward + reward
            state = observation

    avg_run_reward = total_reward / M
    print(avg_run_reward)
    if avg_run_reward > best_result:
        best_weights = new_weights
        best_result = avg_run_reward

print(best_result, best_weights)

print('Play final episode with best model')
env = wrappers.Monitor(env,settings.FSOUTPUT_ENV_VIDEO,force=True)

state = env.reset()
total_reward = 0
done = False
for _ in range(200):
    action = 1 if state.dot(best_weights) > 0 else 0
    new_state, reward, done, info = env.step(action=action)
    total_reward = total_reward + reward
    state = new_state
    if done: break
env.env.close()
env.close()
print('Total reward for optimal params: ',total_reward)