import gym
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import aihub.core.common.settings as settings

import aihub.core.models.deeprl.gym.mountaincar.q_learning_legacy as q_learning
from aihub.core.models.deeprl.gym.mountaincar.q_learning_legacy import plot_cost_to_go, FeatureTransformer, Model, plot_running_avg


class SGDOptimizer():

    def __init__(self, **kwargs):
        self.w = None
        self.lr = 1e-2

    def partial_fit(self, X, Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

#TODO horrible design must be REFACTORED
q_learning.SGDRegressor = SGDOptimizer

# calculate everything up to max[Q(s,a)]
# Ex.
# R(t) + gamma*R(t+1) + ... + (gamma^(n-1))*R(t+n-1) + (gamma^n)*max[Q(s(t+n), a(t+n))]
# def calculate_return_before_prediction(rewards, gamma):
#   ret = 0
#   for r in reversed(rewards[1:]):
#     ret += r + gamma*ret
#   ret += rewards[0]
#   return ret


def play_episode(env, model, eps, gamma, n=5):
    prev_state = env.reset()
    new_state = prev_state
    done = False
    total_reward = 0
    rewards = []
    states = []
    actions = []
    step_idx = 0
    gamma_decay_multiplier = np.array([gamma]*n)**np.arange(n)

    #gym will return done after 200 steps, so step_idx check is needed for old versions only
    while not done and step_idx < 10000:
        action = model.sample_action(prev_state, eps)
        states.append(prev_state)
        actions.append(action)
        new_state, reward, done, info = env.step(action)
        rewards.append(reward)

        if len(rewards) >= n:
            latest_n_states_return = gamma_decay_multiplier.dot(rewards[-n:])
            G = latest_n_states_return + gamma**n*np.max(model.predict(new_state)[0])
            model.update(states[-n],actions[-n],G)

        total_reward+=1
        step_idx+=1
        prev_state = new_state

    if n == 1: rewards, states, actions = [],[],[]
    else: rewards, states, actions = rewards[-n:], states[-n:], actions[-n:]

    goal_achieved = new_state[0] > 0.5
    if goal_achieved:
        while rewards:
            G = gamma_decay_multiplier[:len(rewards)].dot(rewards)
            model.update(states[0],actions[0],G)
            rewards.pop(0), states.pop(0), actions.pop(0)
    else:
        while rewards:
            guess_rewards = rewards + [-1] * (n - len(rewards))
            G = gamma_decay_multiplier.dot(guess_rewards)
            model.update(states[0],actions[0],G)
            rewards.pop(0), states.pop(0), actions.pop(0)
    return total_reward

def main():
    env = gym.make('MountainCar-v0')
    transformer = FeatureTransformer(env)
    model = Model(env, transformer, "constant")
    gamma = 0.99

    N =300
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for idx in range(N):
        eps = 0.1 * 0.97**idx
        totalrewards[idx] = play_episode(env, model, eps, gamma)
        print("episode:", idx, "total reward:", totalrewards[idx])
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())


    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)

    env = gym.wrappers.Monitor(env, os.path.join(settings.FSOUTPUT_ENV_VIDEO, 'mountaincar_ncar'), force=True)
    play_episode(env, model, 0, gamma)

    env.env.close()
    env.close()

main()
