import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import gym
import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')
# wandb.init(name='DQN', project="deep-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--time_steps', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--tau', type=float, default=0.125)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)
parser.add_argument('--train_start', type=int, default=1000)

args = parser.parse_args()


class ActionValueModel:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.eps = args.eps
        self.eps_decay = args.eps_decay
        self.eps_min = args.eps_min

        self.compute_loss = tf.keras.losses.MeanSquareError()
        self.opt = tf.keras.optimizers.Adam(args.lr)
        self.model = self.create_model()

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim * args.time_steps,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_dim)
        ])

    def predict(self, state):
        self.model.predict(state)

    def get_action(self, state, training=True):
        state = state.reshape(1, args.time_steps * self.state_dim)
        self.eps *= self.eps_decay
        self.eps = max(self.eps_min, self.eps)
        eps = self.eps if training else 0.01
        q_value = self.model.predict(state)[0]
        if random.random() < eps:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)
    
    def train(self, state, target):
        state = state.reshape(1, args.time_steps * self.state_dim)
        with tf.GradientTape() as tape:
            logits = self.model(state, training=True)
            assert logits.shape == target.shape
            loss = self.compute_loss(tf.stop_gradient(target), logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))


class Agent:
    def __init__(self,
                 env,
                 memory_cap=1000):
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.stored_states = np.zeros((args.time_steps, self.state_dim))
        self.model = ActionValueModel(self.state_dim, self.action_dim)
        self.target_model = ActionValueModel(self.state_dim, self.action_dim)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.model.set_weight(self.model.model.get_weight())

    def update_states(self, next_state):
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        self.stored_states[-1] = next_state
    
    def reset_stored_state(self):
        self.stored_states = np.zeros((args.time_steps, self.state_dim))

    def put_memory(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        samples = random.sample(self.memory, args.batch_size)
        batch_states = []
        batch_target = []
        for sample in samples:
            state, action, reward, next_state, done = sample
            batch_states.append(state.reshape(args.time_steps * self.state_dim))
            state = state.reshape((1, args.time_steps * self.state_dim))
            target = self.target_model.predict(state)[0]
            train_reward = reward * 0.01
            if done:
                target[action] = train_reward
            else:
                next_state = next_state.reshape((1, args.time_steps * self.state_dim))
                next_q_value = max(self.target_model.predict(next_state))
                target[action] = train_reward + next_q_value * args.gamma
            batch_target.append(target)
            self.model.train(np.array(batch_states), np.array(batch_target))    
     
    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            self.reset_stored_state()
            state = self.env.reset()
            self.update_states(state)
            while not done:
                action = self.model.get_action(self.stored_states)
                next_state, reward, done, _ = self.env.step(action)
                prev_stored_states = self.stored_states
                self.update_states(next_state)
                self.put_memory(prev_stored_states, action, reward, self.stored_states, done)

                if len(self.memory) > args.train_start:
                    self.replay()
                total_reward += reward
            self.update_target_model()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))

def main():
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train()

if __name__ == "__main__":
    main()