#Continuous Mountain Car
#Continuous Policy Gradient
#Update reward structure - bigger actions/bigger penalties, reach goal +100 reward.
#Solved when 90 over last 100 episodes

#Use <hill climbing> instead of G.D.

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
            self.W = tf.Variable(np.zeros(M1,M2).astype(np.float32))
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
    def __init__(self,ft, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_var=[]):
        self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var = \
            ft, D, hidden_layer_sizes_mean, hidden_layer_sizes_var

        #Mean network
        self.mean_layers = []
        M1=D
        for M2 in hidden_layer_sizes_var:
            layer = HiddenLayer(M1,M2)
            self.mean_layers.append(layer)
            M1 = M2

        output_layer = HiddenLayer(M1,1,lambda x: x, use_bias=False, zeros=True)
        self.mean_layers.append(output_layer)

        #Variance network
        self.var_layers = []
        M1=D
        for M1 in hidden_layer_sizes_mean:
            layer = HiddenLayer(M1,M2)
            self.var_layers.append(layer)
            M1=M2

        output_layer = HiddenLayer(M1,1, tf.nn.softplus, use_bias=False, zeros=False)
        self.var_layers.append(output_layer)

        #get params from all layers
        self.params = [item for params in [layer.params for layer in (self.mean_layers + self.var_layers)] for item in params]

        #inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None,D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        def get_output(layers):
            A = self.X
            for layer in layers:
                A = layer.forward(A)
            return tf.reshape(A,[-1])

        #calculate output and cost
        mean = get_output(self.mean_layers)
        var = get_output(self.var_layers)+10e-5 #smoothing

        norm = tf.contrib.distributions.Norm(mean,var)
        self.predict_op = tf.clip_by_value(norm.sample(),-1,1)

        #log_prob = norm.log_prob(self.actions)
        #cost = -tf.reduce_sum(...
        #self.train_op = tf.train.optimizers.AdamOptimizer().minimize(cost)

    def set_session(self, session):
        self.session = session

    def init_vars(self):
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op,feed_dict={self.X:X})

    def sample_action(self,state):
        p = self.predict(state)[0]
        return p

    def copy(self):
        clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var)
        clone.set_session(self.session)
        clone.init_vars()
        clone.copy_from(self)

    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p,q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    #Hill climbing implementation
    def pertub_params(self):
        ops = []
        for p in self.params:
            v = self.session.run(p)
            noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
            if np.random.random() < 0.1:
                op = p.assign(noise)
            else:
                op = p.assign(v+noise)
            ops.append(op)
        self.session.run(ops)

def play_episode(env, pmodel, gamma):
    prev_state = env.reset()
    done = False
    total_reward = 0
    step_idx = 0
    while not done and step_idx < 2000:
        action = pmodel.sample_action(prev_state)
        new_state, reward, done, info = env.step([action])
        total_reward += reward
        step_idx+=1
    return total_reward

def play_multiple_episodes(env, T, pmodel, gamma, print_idx=False):
    total_rewards = np.empty(T)
    for i in range(T):
        total_rewards[i]=play_episode(env,pmodel,gamma)
        if print_idx: print(total_rewards[:(i+1)].mean())
    avg_total = total_rewards.mean()
    print('avg total rewards',avg_total)
    return avg_total

def random_search(env, pmodel, gamma):
    total_rewards = []
    best_avg_totalr = float('-inf')
    best_pmodel = pmodel
    num_episodes_per_param_test = 3
    for t in range(100):
        tmp_model = best_pmodel.copy()
        tmp_model.pertub_params()
        avg_total_rewards = play_multiple_episodes(env,num_episodes_per_param_test, tmp_model, gamma)
        total_rewards.append(avg_total_rewards)
        if avg_total_rewards > best_avg_totalr:
            best_pmodel = total_rewards
            best_avg_totalr = avg_total_rewards
    return total_rewards, best_pmodel

def main():
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env,n_components=100)
    D = ft.dimensions
    pmodel = PolicyModel(ft, D, [], [])
    session = tf.InteractiveSession()
    pmodel.set_session(session)
    pmodel.init_vars()
    gamma = 0.99

    total_rewards, best_model = random_search(env,pmodel,gamma)

    avg_total_rewards = play_multiple_episodes(env,100,pmodel,gamma,True)
    plt.plot(total_rewards)
    plt.show()

main()