
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import gym
from collections import deque

tf.keras.backend.set_floatx('float64')
# the library is called opencv-python so make sure you import that

# add this function and call it every time you get the observation from the environment

import cv2

def preprocess(img):

        resized = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        resized = np.divide(resized, 255)
        resized = np.reshape(resized, resized.shape + (1,))
        return resized
'''
??? can't get this shit to fucking run bc the values of the input don't fit
What do ???kkkk
'''
class Agent():
    def __init__(self, env_inp_shape, actions):
        #Neural Network
        self.model = keras.Sequential()
        self.model.add(keras.layers.Reshape((80,80,3), input_shape=(80,80)))
        self.model.add(keras.layers.Conv2D(16, 3, activation='selu', kernel_initializer='lecun_normal', input_shape=env_inp_shape))
        self.model.add(keras.layers.Dense(32, activation='selu', kernel_initializer='lecun_normal'))
        self.model.add(keras.layers.Dense(32, activation='selu', kernel_initializer='lecun_normal'))
        self.model.add(keras.layers.Dense(actions, activation='linear'))
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.0001))

        self.target_model = keras.Sequential()
        self.target_model.add(keras.layers.Conv2D(16, 3, activation='selu', kernel_initializer='lecun_normal', input_shape=env_inp_shape))
        self.target_model.add(keras.layers.Dense(32, activation='selu', kernel_initializer='lecun_normal', input_shape=env_inp_shape))
        self.target_model.add(keras.layers.Dense(32, activation='selu', kernel_initializer='lecun_normal'))
        self.target_model.add(keras.layers.Dense(actions))
        
        self.target_model.set_weights(self.model.get_weights())


        self.Transitions = deque(maxlen=1000)
        self.batch_size = 32
        self.batchs = 1
        self.gamma = 0.99
        self.decay = 0.9999
        self.epsilon = 1
        self.actions = range(actions)
        self.action_size = actions


    def __call__(self, state):
        # if False:
        if random.random() < self.epsilon:
            pred = random.choice(self.actions)
        else:
            pred = self.model(np.array([state]), training=True)[0]
            print(pred)
            pred = tf.argmax(pred).numpy()
            print(pred)
        return pred


    def train(self):
        indices = np.random.choice(len(self.Transitions), self.batchs*self.batch_size, replace=False)
        for i in range(0, self.batchs*self.batch_size, self.batch_size):
            states, actions, rewards, next_states, dones = zip(*[self.Transitions[x] for x in indices[i:i+self.batch_size]])
            preds           = self.model(np.array(states), training=False)
            next_values     = self.target_model(np.array(next_states))
            #next_values     = self.model(np.array(next_states), training=False)
            max_next_values = tf.reduce_max(next_values, axis=-1)
            max_next_values = rewards + self.gamma * max_next_values * dones
            indx_actions    = tf.concat([tf.expand_dims(tf.range(preds.shape[0]), axis=-1), tf.expand_dims(actions, axis=-1)], axis=-1)

            targets         = tf.tensor_scatter_nd_update(preds, indx_actions, max_next_values)
            self.model.fit(np.array(states), targets, verbose=False)


env = gym.make("Pong-v0")
n_games = 1000

agent = Agent(env_inp_shape=env.observation_space.shape, actions=env.action_space.n)
avg_reward = None
steps = 1

for episode in range(n_games):
    ep_reward = 0
    state = env.reset()

    done = False

    for game_step in range(200):
        env.render()

        action = agent(state)
        '''
        Resize, get to grayscale, !get conv layer 
        '''
        next_state, reward, done, _ = env.step(action)
        reward = -1 if done else reward
        
        state = preprocess(state)
        next_state = preprocess(next_state)
        

        agent.Transitions.append([state, action, reward, next_state, int(not done)])
        if len(agent.Transitions) > agent.batchs*agent.batch_size:
            if steps%1000==0:
                agent.target_model.set_weights(agent.model.get_weights())
                print('updated')
            agent.train()
            agent.epsilon *= agent.decay if agent.epsilon > 0.05 else 1

        ep_reward += reward
        steps+=1

        state=next_state

        if done:
            break
    avg_reward = ep_reward if avg_reward == None else avg_reward * 0.99 + ep_reward * 0.01
    print('\nEpisode: %d | Episode Reward: %i | Average Reward: %f | Epsilon: %f' % (episode, ep_reward, avg_reward, agent.epsilon))

print('Done:', done)