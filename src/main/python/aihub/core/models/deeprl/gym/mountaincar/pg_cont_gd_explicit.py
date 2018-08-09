#Continuous Mountain Car
#Continuous Policy Gradient
#Update reward structure - bigger actions/bigger penalties, reach goal +100 reward.
#Solved when 90 over last 100 episodes

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from aihub.core.models.deeprl.gym.mountaincar.q_learning_legacy import FeatureTransformer

#TODO - code below primarily reflects HW of DeepRL course
# There are plenty ways to improve and refactor, which will be done in separate model
# - refactor, to get rid of manual layer definitions
# - use builtin initializers
# - create NN that combines Policy and Value approximators
# - copy/copy_from methods must follow clone pattern

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=False, zeros=False):
        if zeros:
            self.W = tf.Variable(np.zeros((M1,M2)).astype(np.float32))
        else:
            self.W = tf.Variable(tf.random_normal(shape=(M1,M2)))

        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b=tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f=f

    def forward(self, X):
        if self.use_bias: z = tf.matmul(X, self.W)+self.b
        else: z = tf.matmul(X,self.W)
        return self.f(z)

class PolicyModel:
    def __init__(self,ft, D, hidden_layer_sizes=[]):
        self.ft, self.D, self.hidden_layer_sizes= ft, D, hidden_layer_sizes

        self.layers = []
        M1=D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1 = M2

        self.mean_layer = HiddenLayer(M1,1,lambda x: x, use_bias=False, zeros=True)

        self.stdv_layer = HiddenLayer(M1,1, tf.nn.softplus, use_bias=False, zeros=False)

        #get params from all layers
        # self.params = [item for params in [layer.params for layer in (self.mean_layers + self.var_layers)] for item in params]

        #inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None,D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        A = self.X
        for layer in self.layers:
            A = layer.forward(A)

        #calculate output and cost
        mean = self.mean_layer.forward(A)
        stdv = self.stdv_layer.forward(A)+10e-5 #smoothing

        norm = tf.contrib.distributions.Normal(mean,stdv)
        self.predict_op = tf.clip_by_value(norm.sample(),-1,1)

        log_probs = norm.log_prob(self.actions)
        cost = -tf.reduce_sum(self.advantages*log_probs+0.1*norm.entropy())
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)


    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)

        self.session.run(self.train_op, feed_dict={self.X: X, self.actions: actions, self.advantages: advantages})

    # def init_vars(self):
    #     init_op = tf.variables_initializer(self.params)
    #     self.session.run(init_op)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op,feed_dict={self.X:X})

    def sample_action(self,state):
        p = self.predict(state)[0]
        return p

    # def copy(self):
    #     clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes)
    #     clone.set_session(self.session)
    #     clone.init_vars()
    #     clone.copy_from(self)
    #
    # def copy_from(self, other):
    #     ops = []
    #     my_params = self.params
    #     other_params = other.params
    #     for p,q in zip(my_params, other_params):
    #         actual = self.session.run(q)
    #         op = p.assign(actual)
    #         ops.append(op)
    #     self.session.run(ops)
    #
    # #Hill climbing implementation
    # def pertub_params(self):
    #     ops = []
    #     for p in self.params:
    #         v = self.session.run(p)
    #         noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
    #         if np.random.random() < 0.1:
    #             op = p.assign(noise)
    #         else:
    #             op = p.assign(v+noise)
    #         ops.append(op)
    #     self.session.run(ops)

class ValueModel:
    def __init__(self, ft, D, hidden_layer_sizes=[]):
        self.ft = ft
        self.costs = []
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            self.layers.append(HiddenLayer(M1,M2))
            M1 = M2

        self.layers.append(HiddenLayer(M1,1,lambda x: x))
        self.X = tf.placeholder(tf.float32, shape=(None,D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None), name='Y')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)

        Y_hat = tf.reshape(Z,[-1])
        self.predict_op = Y_hat
        self.cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.train_op = tf.train.AdadeltaOptimizer(1e-1).minimize(self.cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op,feed_dict={self.X:X, self.Y:Y})
        cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y})
        self.costs.append(cost)

    def predict(self, X):
        X = self.ft.transform(np.atleast_2d(X))
        return self.session.run(self.predict_op, feed_dict={self.X:X})

def play_episode_td(env,pmodel,vmodel,gamma):
    prev_state = env.reset()
    done = False
    step_idx = 0
    total_reward = 0

    while not done and step_idx<2000:
        action = pmodel.sample_action(prev_state)
        new_state, reward, done, info = env.step(action)
        total_reward += reward

        V_next = vmodel.predict(new_state)
        G = reward + gamma*V_next
        advantage = G - vmodel.predict(prev_state)
        pmodel.partial_fit(prev_state,action,advantage)
        vmodel.partial_fit(prev_state,G)
        step_idx+=1
    return total_reward, step_idx

#
# def play_episode(env, pmodel, gamma):
#     prev_state = env.reset()
#     done = False
#     total_reward = 0
#     step_idx = 0
#     while not done and step_idx < 2000:
#         action = pmodel.sample_action(prev_state)
#         new_state, reward, done, info = env.step([action])
#         total_reward += reward
#         step_idx+=1
#     return total_reward
#
# def play_multiple_episodes(env, T, pmodel, gamma, print_idx=False):
#     total_rewards = np.empty(T)
#     for i in range(T):
#         total_rewards[i]=play_episode(env,pmodel,gamma)
#         if print_idx: print(total_rewards[:(i+1)].mean())
#     avg_total = total_rewards.mean()
#     print('avg total rewards',avg_total)
#     return avg_total
#
# def random_search(env, pmodel, gamma):
#     total_rewards = []
#     best_avg_totalr = float('-inf')
#     best_pmodel = pmodel
#     num_episodes_per_param_test = 3
#     for t in range(100):
#         tmp_model = best_pmodel.copy()
#         tmp_model.pertub_params()
#         avg_total_rewards = play_multiple_episodes(env,num_episodes_per_param_test, tmp_model, gamma)
#         total_rewards.append(avg_total_rewards)
#         if avg_total_rewards > best_avg_totalr:
#             best_pmodel = total_rewards
#             best_avg_totalr = avg_total_rewards
#     return total_rewards, best_pmodel

def main():
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env,n_components=100)
    D = ft.dimensions
    pmodel = PolicyModel(ft, D, [])
    vmodel = ValueModel(ft,D,[])
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.95

    N = 50
    total_rewards = np.empty(N)
    for idx in range(N):
        total_rewards[idx], num_steps = play_episode_td(env,pmodel,vmodel,gamma)
        print('Reward: <{}> with steps: <{}>'.format(total_rewards[idx],num_steps))
    plt.plot(total_rewards)
    plt.show()

main()