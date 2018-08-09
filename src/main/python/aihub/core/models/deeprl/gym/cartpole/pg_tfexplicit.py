import gym
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime as dt

from aihub.core.models.deeprl.gym.mountaincar.q_learning_legacy import plot_cost_to_go, plot_running_avg
from aihub.core.common import settings


class HiddenLayer:
    def __init__(self, N_l_input, N_l_output, f=tf.nn.tanh,use_bias=True):
        self.W = tf.Variable(tf.random_normal(shape=(N_l_input,N_l_output)))
        self.use_bias = use_bias
        if use_bias: self.b = tf.Variable(np.zeros(N_l_output).astype(np.float32))
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X,self.W)+self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)

#approximates Pi(a|s)
class PolicyModel:
    def __init__(self, D, K, hidden_layer_sizes):
        self.layers=[]
        #create nn model,
        #K - number of actions
        N_l_input = D
        for N_l_output in hidden_layer_sizes:
            self.layers.append(HiddenLayer(N_l_input,N_l_output))
            N_l_input = N_l_output

        output_layer = HiddenLayer(N_l_input, K, f=tf.nn.softmax, use_bias=False)
        self.layers.append(output_layer)

        self.X = tf.placeholder(tf.float32, shape=(None,D), name='X')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,),name='advantages')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)

        p_a_given_s = Z
        #action scores = Z

        self.predict_op = p_a_given_s

        selected_probs = tf.log(tf.reduce_sum(p_a_given_s * tf.one_hot(self.actions, K), reduction_indices = [1] ))

        cost = -tf.reduce_sum(self.advantages*selected_probs)

        self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(self.train_op,feed_dict={self.X:X, self.actions:actions,self.advantages: advantages})

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X:X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return np.random.choice(len(p),p=p) #Model "betterness" directly instead of epsilon-greedy model

#Model for V(s) approximation
class ValueModel:
    def __init__(self, D, hidden_layer_sizes):
        #create tf model
        self.layers=[]
        #create nn model,
        #K - number of actions
        N_l_input = D
        for N_l_output in hidden_layer_sizes:
            self.layers.append(HiddenLayer(N_l_input,N_l_output))
            N_l_input = N_l_output

        output_layer = HiddenLayer(N_l_input, 1, f=lambda x:x)
        self.layers.append(output_layer)

        self.X = tf.placeholder(tf.float32, shape=(None,D),name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,),name='Y')

        #Output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z,[-1])

        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y-Y_hat))
        #self.train_op = tf.train.GradientDescentOptimizer(10e-5).minimize(cost)
        #self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)
        self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.X:X, self.Y:Y})

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X:X})

def play_episode_td(env, pmodel, vmodel, gamma):
    prev_state = env.reset()
    done = False
    total_rewards = 0
    step_idx = 0
    while not done and step_idx < 1000:
        action = pmodel.sample_action(prev_state)
    #TODO

def play_episode_mc(env,pmodel, vmodel, gamma):
    prev_state = env.reset()
    reward = 0
    done = False
    total_rewards = 0
    step_idx = 0

    states, rewards, actions = [], [], []

    while not done and step_idx < 2000:
        action = pmodel.sample_action(prev_state)
        states, rewards, actions = states + [prev_state], rewards + [reward], actions + [action]

        new_state, reward, done, info = env.step(action)
        #if done and step_idx < 200: reward = -200
        if done: reward = -200

        if reward == 1: total_rewards += reward
        prev_state = new_state
        step_idx+=1

    action = pmodel.sample_action(prev_state)
    states.append(prev_state)
    actions.append(action)
    rewards.append(reward)

    returns, advantages, G = [], [], 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G-vmodel.predict(s)[0])
        G = r+gamma*G
    returns.reverse()
    advantages.reverse()

    pmodel.partial_fit(states, actions, advantages)
    vmodel.partial_fit(states, returns)

    return total_rewards

def main():
    env = gym.make('CartPole-v0')
    D = env.observation_space.shape[0]
    K = env.action_space.n
    pmodel = PolicyModel(D, K, [])
    vmodel = ValueModel(D, [10])
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.99
    N = 2000
    total_rewards = np.empty(N)
    costs = np.empty(N)
    for idx in range(N):
        total_rewards[idx] = play_episode_mc(env, pmodel, vmodel, gamma)
        if idx % 100 == 0:
            print("episode:", idx, "total reward:", total_rewards[idx], "avg reward (last 100):")
            #, total_rewards[-min(len(total_rewards),100):].mean()

    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
    print("total steps:", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)


    env = gym.wrappers.Monitor(env,os.path.join(settings.FSOUTPUT_ENV_VIDEO,'gym_cartpole_pg_tf_explicit'),force=True)
    play_episode_mc(env, pmodel, vmodel, gamma)
    env.env.close()
    env.close()

main()
