import gym
import matplotlib.pyplot as plt
from scipy.misc import imresize
import tensorflow as tf
import numpy as np

IM_WIDTH = IM_HEIGHT = 80

def downsample_image(A):
    B = A[31:195]
    B = B.mean(axis=2) / 255.0
    C = imresize(B, size=(IM_WIDTH, IM_HEIGHT), interp='nearest')
    return C

class DQN:
    experiences = []
    min_experiences = 100
    max_experiences = 10000

    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, scope):
        self.K = K
        self.scope = scope

        with tf.variable_scope(scope):

            self.X = tf.placeholder(tf.float32, shape=(None,4,IM_WIDTH,IM_HEIGHT), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')


            Z = self.X
            Z = tf.transpose(Z,[0,2,3,1])
            for num_output_filters, filter_size, pool_size in conv_layer_sizes:
                Z = tf.contrib.layers.conv2d(Z,num_output_filters, filter_size, pool_size, activation_fn = tf.nn.relu)

            Z = tf.contrib.layers.flatten(Z)
            for M in hidden_layer_sizes:
                Z = tf.contrib.layers.fully_connected(Z,M)

            self.predict_op = tf.contrib.layers.fully_connected(Z,K)

            selected_action_values = tf.reduce_sum( self.predict_op * tf.one_hot(self.actions, K), reduction_indices=[1])

            self.cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
            self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)

    def set_session(self,session):
        self.session = session

    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v:v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        ops = []
        for p, q in zip(mine, theirs):
            actual = self.session.run(p)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def predict(self,X):
        return self.session.run(self.predict_op, feed_dict={self.X:X})

    def train(self, target_network):
        if len(self.experiences) < self.min_experiences: return
        sample = np.random.sample
        #TODO


    def add_experience(self, s, a, r, s_):
        if len(self.experiences) > self.max_experiences: self.experiences.pop(0)
        self.experiences.append((s,a,r,s_))

    def sample_action(self, state, eps):
        if np.random.rand() < eps: return np.random.choice(self.K)
        return np.argmax(self.predict(state)[0])

def update_state(state, observation):
    state.append(downsample_image(observation))
    if len(state)>4: state.pop(0)

def play_episode(env, model, tmodel, eps, eps_step, gamma, copy_period):
    observation = env.reset()
    done = False
    total_reward = 0
    step_idx = 0
    state = []
    prev_state = []
    update_state(state,observation)
    while not done and step_idx < 2000:
        if len(state) < 4: action = env.action_space.sample() #model.sample_action(state,1.0 if len(state) < 4 else eps)
        else: action = model.sample_action(state,eps)
        prev_state.append(state[-1])
        if len(prev_state) > 4: prev_state.pop(0)

        observation, reward, done, info = env.step(action)
        update_state(state,observation)
        total_reward += reward
        if done: reward -=200

        model.add_experience(prev_state,action,reward,state)
        model.train(tmodel)

        if step_idx % copy_period == 0: tmodel.copy_from(model)

        step_idx +=1
        eps = max(eps - eps_step, 0.1)

    return total_reward, eps, step_idx

def main():
    env = gym.make('Breakout-v0')
    gamma = 0.99
    copy_period = 10000

plt.imshow(A)
plt.show()
plt.imshow(downsample_image(A))
plt.show()