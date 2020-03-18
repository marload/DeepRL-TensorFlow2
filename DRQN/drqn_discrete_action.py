import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Lambda

import gym
import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')
wandb.init(name='DRQN', project="deep-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--time_steps', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--tau', type=float, default=0.125)

args = parser.parse_args()


class Agent:
    def __init__(self,
                 env,
                 memory_cap=1000,
                 eps=1.0,
                 eps_decay=0.995,
                 eps_min=0.01):
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.stored_states = np.zeros((args.time_steps, self.state_dim))

        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(args.lr)
        self.model = self.create_model()
        self.t_model = self.create_model()
        self.t_model.set_weights(self.model.get_weights())

    def create_model(self):
        return tf.keras.Sequential([
            Input((args.time_steps, self.state_dim)),
            LSTM(128, activation='tanh'),
            Dense(128, activation='relu'),
            Dense(self.action_dim)
        ])

    def get_action(self, training=True):
        states = np.expand_dims(self.stored_states, axis=0)
        self.eps *= self.eps_decay
        self.eps = max(self.eps_min, self.eps)
        eps = self.eps if training else 0.01
        q_values = self.model.predict(states)[0]
        if np.random.normal() < eps:
            return self.env.action_space.sample()
        return np.argmax(q_values)

    def put_memory(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def update_states(self, next_state):
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        self.stored_states[-1] = next_state

    def target_update(self):
        weights = self.model.get_weights()
        t_weights = self.t_model.get_weights()
        for i in range(len(t_weights)):
            t_weights[i] = weights[i] * args.tau + \
                t_weights[i] * (1 - args.tau)
        self.t_model.set_weights(t_weights)

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            assert logits.shape == y.shape
            loss = self.compute_loss(tf.stop_gradient(y), logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def replay(self):
        samples = random.sample(self.memory, args.batch_size)
        batch_states = []
        batch_target = []
        for sample in samples:
            states, action, reward, next_states, done = sample
            batch_states.append(states)
            states = np.expand_dims(states, axis=0)
            target = self.t_model.predict(states)[0]
            train_reward = reward * 0.01
            if done:
                target[action] = train_reward
            else:
                next_states = np.expand_dims(next_states, axis=0)
                next_q_value = max(self.t_model.predict(next_states)[0])
                target[action] = train_reward + next_q_value * args.gamma
            batch_target.append(target)
        self.train_step(np.array(batch_states), np.array(batch_target))

    def train(self, max_episodes=1000):
        done, ep, total_reward = True, 0, 0
        while ep <= max_episodes:
            if done:
                self.stored_states = np.zeros(
                    (args.time_steps, self.state_dim))
                print('EP{} EpisodeReward={}'.format(ep, total_reward))
                wandb.log({'Reward': total_reward})

                done, cur_state, total_reward = False, self.env.reset(), 0
                self.update_states(cur_state)
                ep += 1

            action = self.get_action()
            next_state, reward, done, _ = self.env.step(action)
            prev_stored_states = self.stored_states
            self.update_states(next_state)
            self.put_memory(prev_stored_states, action,
                            reward, self.stored_states, done)

            if len(self.memory) >= args.batch_size:
                self.replay()
            self.target_update()

            total_reward += reward


def main():
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train(max_episodes=1000)


if __name__ == "__main__":
    main()
