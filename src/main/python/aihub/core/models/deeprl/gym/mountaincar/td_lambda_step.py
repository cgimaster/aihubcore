import gym
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import aihub.core.common.settings as settings

import aihub.core.models.deeprl.gym.mountaincar.q_learning_legacy as q_learning
from aihub.core.models.deeprl.gym.mountaincar.q_learning_legacy import plot_cost_to_go, FeatureTransformer, plot_running_avg


class BaseModel:

    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, inp, target, eligibility, lr=1e-2):
        self.w += lr * (target - inp.dot(self.w))*eligibility

    def predict(self, X):
        return np.array(X).dot(self.w)


class TDLyabmdaModel:
    def __init__(self, env, transformer):
        self.env = env
        self.transformer = transformer
        D = transformer.dimensions

        self.eligibilities = np.zeros((env.action_space.n,D))
        self.models = [BaseModel(D) for _ in range(env.action_space.n)]

    def predict(self, state):
        X = self.transformer.transform([state])
        assert len(X.shape) == 2
        #return np.array([m.predict(X)[0] for m in self.models])
        return np.stack([m.predict(X) for m in self.models])

    def update(self, state, action, G, gamma, lyambda):
        X = self.transformer.transform([state])
        self.eligibilities *= gamma*lyambda
        #for linear model W*X+B, dW = X - for eligibilities where action is taken.
        #rest is 0
        eligibility_gradient = X[0]
        self.eligibilities[action] += eligibility_gradient
        self.models[action].partial_fit(X[0],G,self.eligibilities[action])

    def sample_action(self, state, eps):
        if np.random.rand() < eps: return self.env.action_space.sample()
        else: return np.argmax(self.predict(state))


def play_episode(model, env, eps, gamma, lyambda):
    done = False
    prev_state = env.reset()
    total_rewards = 0
    step_idx = 0
    while not done and step_idx < 1000:
        action = model.sample_action(prev_state, eps)
        new_state, reward, done, info = env.step(action)
        next_action = model.predict(new_state)[0]
        G = reward + gamma*np.max(next_action)
        model.update(prev_state,action,G,gamma,lyambda)
        prev_state, step_idx, total_rewards = new_state, step_idx+1, total_rewards+reward
    return total_rewards


def main():
    env = gym.make('MountainCar-v0')
    transformer = FeatureTransformer(env)
    model = TDLyabmdaModel(env, transformer)
    gamma = 0.9999
    lyambda = 0.7
    N = 1000
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for idx in range(N):
        eps = 0.1 * (0.97**idx)
        totalrewards[idx] = play_episode(model, env, eps, gamma, lyambda=lyambda)
        print("episode:", idx, "total reward:", totalrewards[idx])
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())


    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)

    env = gym.wrappers.Monitor(env, os.path.join(settings.FSOUTPUT_ENV_VIDEO, 'mountaincar_tdlyambda'), force=True)
    play_episode(model, env, 0, gamma, lyambda)

    env.env.close()
    env.close()

main()
