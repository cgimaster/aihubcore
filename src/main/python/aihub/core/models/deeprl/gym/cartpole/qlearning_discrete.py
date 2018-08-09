import gym
import numpy as np
from aihub.core.common import settings
import os
import matplotlib.pyplot as plt

np.random.seed(1)

#convert discretized state to index
def convert(bins):
    return int("".join([str(int(k)) for k in bins]))

class StateTransformer():
    cart_position_bins = np.linspace(-2.4, 2.4, 9)
    cart_velocity_bins = np.linspace(-2, 2, 9) # (-inf, inf)
    pole_angle_bins = np.linspace(-0.4, 0.4, 9)
    pole_velocity_bins = np.linspace(-3.5, 3.5, 9) # (-inf, inf)

    dimensions_bins = [cart_position_bins,cart_velocity_bins,pole_angle_bins,pole_velocity_bins]

    def transform_state(self,state):
        bins = [np.digitize([state[idx]], self.dimensions_bins[idx])[0] for idx in range(state.shape[0])]
        return convert(bins)

transformer = StateTransformer()

def initialize_Q():
    total_states = 10**4
    actions = 2
    Q = np.random.rand(total_states,actions)
    return Q

def sample_action(Q, state, epsilon):
    Qs = Q[transformer.transform_state(state)]
    if np.random.rand() < epsilon:
        result = 1 if np.random.rand() > 0.5 else 0
    else:
        result = np.argmax(Qs)
    return result

def play_episode(env, Q, epsilon, alpha=0.01, gamma=0.5, fail_reward=-300):
    prev_state = env.reset()
    done = False
    t = 0
    total_reward = 0
    while not done and t < 200:
        t += 1
        action = sample_action(Q, prev_state, epsilon)
        new_state, reward, done, info = env.step(action = action)
        Qs_idx = transformer.transform_state(prev_state)
        if done and t < 200: #failed state
            reward = fail_reward

        G = reward + gamma * np.max(Q[transformer.transform_state(new_state)])
        total_reward += 1
        Q[Qs_idx][action] = Q[Qs_idx][action] + alpha * (G - Q[Qs_idx][action])
        prev_state = new_state
    return total_reward

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


env = gym.make('CartPole-v0')

episodes_N = 50000
historical_rewards = np.empty(episodes_N)
Q = initialize_Q()
for episode_idx in range(episodes_N):
    eps = 1.0/np.sqrt(episode_idx+1)
    episode_reward = play_episode(env, Q,epsilon=eps)
    historical_rewards[episode_idx] = episode_reward
    if episode_idx % 100 == 0:
        print("episode:", episode_idx, "episode reward:", episode_reward, "eps:", eps)
print("avg reward for last 100 episodes:", historical_rewards[-100:].mean())
print("total steps:", historical_rewards.sum())

env = gym.wrappers.Monitor(env,os.path.join(settings.FSOUTPUT_ENV_VIDEO,'gym_cartpole_qlearning_discrete'),force=True)
play_episode(env,Q,epsilon=0)
env.env.close()
env.close()

plt.plot(historical_rewards)
plt.title("Rewards")
plt.show()
plot_running_avg(historical_rewards)
