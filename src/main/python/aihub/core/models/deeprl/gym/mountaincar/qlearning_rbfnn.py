import gym
import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import os
import aihub.core.common.settings as settings
import matplotlib

class FeatureTransformer():
    def __init__(self, env, n_components = 500, n_samples = 10000):
        observation_examples = np.array([env.observation_space.sample() for x in range(n_samples)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        #RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
                ])

        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        result = self.featurizer.transform(scaled)
        return result

class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate = learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    def predict(self, state):
        X = self.feature_transformer.transform([state])
        result = np.stack([m.predict(X) for m in self.models]).T
        return result

    def update(self, state, action, G):
        X = self.feature_transformer.transform([state])
        self.models[action].partial_fit(X, [G])

    def sample_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(state))


def play_episode(env, model, epsilon, gamma = 0.99, max_steps=1000, fail_reward = -300):
    state = env.reset()
    done = False
    total_reward = 0
    idx = 0
    while not done and idx<max_steps:
        idx+=1
        action = model.sample_action(state,epsilon)
        new_state, reward, done, info = env.step(action)
        #if not done and idx<max_steps: reward = fail_reward
        G = reward + gamma*np.max(model.predict(new_state)[0])
        model.update(state,action,G)
        state = new_state
        total_reward += reward
    return total_reward



def plot_cost_to_go(env, estimator, num_tiles=20):
  x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
  y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
  X, Y = np.meshgrid(x, y)
  # both X and Y will be of shape (num_tiles, num_tiles)
  Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
  # Z will also be of shape (num_tiles, num_tiles)

  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
  ax.set_xlabel('Position')
  ax.set_ylabel('Velocity')
  ax.set_zlabel('Cost-To-Go == -V(s)')
  ax.set_title("Cost-To-Go Function")
  fig.colorbar(surf)
  plt.show()


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


def main():
    env = gym.make('MountainCar-v0')
    transformer = FeatureTransformer(env)
    model = Model(env,transformer,learning_rate="constant")

    N = 500
    total_rewards = np.empty(N)
    for eposide_idx in range(N):
        epsilon = 0.1*0.997**eposide_idx
        #epsilon = 1.0/(1+eposide_idx)
        gamma = 0.9995
        total_rewards[eposide_idx] = play_episode(env,model,epsilon,gamma=gamma)
        if eposide_idx % 100 == 0: print("episode:", eposide_idx, "total reward:", total_rewards[eposide_idx], epsilon)
    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
    print("total steps:", -total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()
    plot_running_avg(total_rewards)

    plot_cost_to_go(env, model)
    env = gym.wrappers.Monitor(env, os.path.join(settings.FSOUTPUT_ENV_VIDEO, "mountaincar_qlearning_rbfnn"), force=True)
    play_episode(env, model, 0)
    env.env.close()
    env.close()

if __name__ == '__main__':
    main()