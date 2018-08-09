import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from aihub.core.common import settings
import os

from aihub.core.models.deeprl.gym.mountaincar.q_learning_legacy import FeatureTransformer, plot_running_avg

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        self.W=tf.Variable(tf.random_normal(shape=(M1,M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f=f

    def forward(self,X):
        if self.use_bias:
            a = tf.matmul(X,self.W) + self.b
        else:
            a = tf.matmul(X,self.W)
        return self.f(a)

class DQN:
    def __init__(self, D, K, hidden_layer_sizes,gamma,
                        max_experiences=10000, min_experiences=100, batch_size=32):
        self.D, self.K, self.hidden_layer_sizes, self.gamma, self.max_experiences,  self.min_experiences, self.batch_size = \
            D, K, hidden_layer_sizes, gamma, max_experiences, min_experiences, batch_size

        self.layers = []

        M1=D
        for M2 in hidden_layer_sizes:
            self.layers.append(HiddenLayer(M1,M2))
            M1=M2

        self.layers.append(HiddenLayer(M1,K,f=lambda x:x))

        self.params = []
        for layer in self.layers:
            self.params += layer.params

        self.X = tf.placeholder(tf.float32, shape=(None,D), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(dtype=tf.int32, shape=(None,), name='actions')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        selected_action_values = tf.reduce_sum(
            Y_hat * tf.one_hot(self.actions, K),
            reduction_indices=[1]
        )
        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdamOptimizer(10e-3).minimize(cost)

        #replay memory
        self.experiences = {'s':[], 'a':[], 'r':[], 's_':[]}

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        ops = []
        for p, q in zip(self.params, other.params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X:X})

    def train(self, target_network):
        if len(self.experiences['s']) < self.min_experiences: return

        idx = np.random.choice(len(self.experiences['s']), size=self.batch_size, replace=False)

        states = [self.experiences['s'][i] for i in idx]
        actions = [self.experiences['a'][i] for i in idx]
        rewards = [self.experiences['r'][i] for i in idx]
        next_states = [self.experiences['s_'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = [r+self.gamma*next_q for r, next_q in zip(rewards,next_Q)]

        self.session.run(self.train_op, feed_dict={self.X:states,self.G:targets, self.actions:actions})

    def add_experience(self, s, a, r, s_):
        if len(self.experiences['s']) > self.max_experiences:
            for k in self.experiences: self.experiences[k].pop(0)
        self.experiences['s'].append(s)
        self.experiences['a'].append(a)
        self.experiences['r'].append(r)
        self.experiences['s_'].append(s_)

    def sample_action(self, X, eps):
        if np.random.random() < eps: return np.random.choice(self.K)
        X = np.atleast_2d(X)
        return np.argmax(self.predict(X)[0])

def play_episode(env, model, tmodel, eps, gamma, copy_period):
    prev_state = env.reset()
    done = False
    total_reward = 0
    step_idx = 0
    while not done and step_idx < 2000:
        step_idx +=1
        action = model.sample_action(prev_state, eps)
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        # if done and step_idx < 200:
        #     reward = -200 #This is most likely error with new version of gym
        if done:
            reward = -200

        model.add_experience(prev_state, action, reward, new_state)
        model.train(tmodel)
        if step_idx % copy_period == 0: tmodel.copy_from(model)
        prev_state = new_state
    return total_reward

def main():
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_period = 50
    D = len(env.observation_space.sample())
    K = env.action_space.n
    sizes = [200,200]
    model = DQN(D,K,sizes,gamma)
    tmodel = DQN(D, K, sizes, gamma) #why not tmodel.copy_from(model)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    model.set_session(session)
    tmodel.set_session(session)


    N = 2000
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_episode(env, model, tmodel, eps, gamma, copy_period)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    env = gym.wrappers.Monitor(env,os.path.join(settings.FSOUTPUT_ENV_VIDEO,'gym_cartpole_dqn_tf_explicit'),force=True)
    play_episode(env, model, tmodel, 0, gamma, 1000)
    env.env.close()
    env.close()

main()