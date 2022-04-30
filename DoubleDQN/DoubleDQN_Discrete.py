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
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--episode', type=int, default=32)
parser.add_argument('--replay_batch', type=int, default=10)
parser.add_argument('--sample_method', type=str, default="recent")
parser.add_argument('--train_policy', type=str, default="off-policy")
parser.add_argument('--eps', type=float, default=0.9)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()


class ReplayBuffer:
  def __init__(self, capacity=64):
    self.buffer = deque(maxlen=capacity)
    self.sample_policy = args.sample_method

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
    states = states.reshape(bs, -1)
    next_states = next_states.reshape(bs, -1)
    done = done.reshape(bs, 1)
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

  def get_action(self, state, deterministic=True):
    self.epsilon *= args.eps_decay
    self.epsilon = max(self.epsilon, args.eps_min)
    q_value = self.predict(state)[0]
    if not deterministic:
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

  def vec_reward(self, a, value):
    reward = np.ones((args.batch_size, self.action_dim)) * Maze.NEU_REWARD
    row, col = list(range(args.batch_size)), a
    reward[row, col] = value
    # reward = np.exp(reward)
    # reward = reward/np.sum(reward, axis=1, keepdims=True)
    return reward

  def replay(self):
    for _ in range(args.replay_batch):
      if self.buffer.size() < args.batch_size:
        break
      states, actions, rewards, next_states, _ = self.buffer.sample()
      rewards = self.vec_reward(actions, rewards)
      next_q_values = self.target_model.predict(next_states)
      targets = rewards + next_q_values * args.gamma
      self.model.train(states, targets)

  def train(self):
    if args.train_policy == "on-policy":
      self.train_onpolicy()
    elif args.train_policy == "off-policy":
      self.train_offpolicy()

  def train_offpolicy(self):
    for ep in range(args.episode):
      self.env.reset()
      for _ in range(args.batch_size):
        state = self.env.rand_state()
        for action in range(self.action_dim):
          next_state, reward, done = self.env.action(action)
          self.buffer.put(state, action, reward, next_state, done)
      self.replay()
      self.target_update()

      for rob, s in self.env.iter_states():
        act = self.model.get_action(s)
        print(f'      at {rob}, action = {act}')
      done, total_reward = False, 0
      state = self.env.reset()
      step = 0
      while not done and step < 10:
        self.env.print()
        action = self.model.get_action(state)
        print(f"action: {action}")
        next_state, reward, done = self.env.action(action)
        total_reward += reward
        state = next_state
        step += 1
      print(f'Episode {ep}, action rewards:{total_reward}')


  def train_onpolicy(self):
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
