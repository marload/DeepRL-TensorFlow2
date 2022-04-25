from MazeEnv import Maze
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam

import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--episode', type=int, default=32)
parser.add_argument('--replay_batch', type=int, default=10)
parser.add_argument('--eps', type=float, default=0.9)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()


class ReplayBuffer:
  def __init__(self, capacity=64, sample_policy='recent'):
    self.buffer = deque(maxlen=capacity)
    self.sample_policy = sample_policy

  def put(self, state, action, reward, next_state, done):
    self.buffer.append([state, action, reward, next_state, done])

  def sample(self):
    bs = args.batch_size
    if self.sample_policy == 'random':
      sample = random.sample(self.buffer, args.batch_size)
    if self.sample_policy == 'recent':
      sample = list(map(lambda x: self.buffer.pop(), range(bs)))
      if sample is None:
        return [None]*5
    states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
    states = np.array(states).reshape(bs, -1)
    next_states = np.array(next_states).reshape(bs, -1)
    return states, actions, rewards, next_states, done


  def size(self):
    return len(self.buffer)


class QualityModel:
  def __init__(self, state_dim, aciton_dim):
    self.state_dim = state_dim
    self.action_dim = aciton_dim
    self.epsilon = args.eps
    self.updates = 0
    self.model = self.create_model()

  def create_model(self):
    model = tf.keras.Sequential([
      Input((self.state_dim,)),
      Dense(32, activation='relu'),
      Dense(16, activation='relu'),
      Dense(self.action_dim)
    ])
    model.compile(loss='mse', optimizer=Adam(args.lr))
    return model

  def predict(self, state):
    return self.model.predict(state)

  def get_action(self, state):
    self.epsilon *= args.eps_decay
    self.epsilon = max(self.epsilon, args.eps_min)
    q_value = self.predict(state)[0]
    if np.random.random() < self.epsilon:
      return random.randint(0, self.action_dim - 1)
    return np.argmax(q_value)

  def train(self, states, targets):
    self.updates += 1
    self.model.fit(states, targets, epochs=1, verbose=0)


class Agent:
  def __init__(self, env:Maze):
    self.env = env
    self.buffer = ReplayBuffer(args.batch_size*2)
    self.state_dim = self.env.state_dim()
    self.action_dim = self.env.action_dim()

    self.model = QualityModel(self.state_dim, self.action_dim)
    self.target_model = QualityModel(self.state_dim, self.action_dim)
    self.target_update()


  def target_update(self):
    weights = self.model.model.get_weights()
    self.target_model.model.set_weights(weights)

  def replay(self):
    for _ in range(args.replay_batch):
      if self.buffer.size() < args.batch_size:
        break
      states, actions, rewards, next_states, done = self.buffer.sample()
      targets = self.target_model.predict(states)
      next_q_values = self.target_model.predict(next_states)[:, np.argmax(self.model.predict(next_states), axis=1)]
      targets[:, actions] = rewards + (1 - done) * next_q_values * args.gamma
      self.model.train(states, targets)

  def train(self):
    for ep in range(args.episode):
      done, total_reward = False, 0
      state = self.env.reset()
      while not done:
        self.env.print()
        action = self.model.get_action(state)
        next_state, reward, done = self.env.action(action)
        self.buffer.put(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

      if self.buffer.size() >= args.batch_size:
        self.replay()
      self.target_update()
      print('EP{} EpisodeReward={}, reward={}, updates = {}'.format(ep, total_reward, reward, self.model.updates))


def main():
  print(f"args = {args}")
  env = Maze(5)
  agent = Agent(env)
  agent.train()


if __name__ == "__main__":
  main()
