from MazeEnv import Maze
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--rand_seed', type=int, default=None)
parser.add_argument('--episode', type=int, default=32)
parser.add_argument('--replay_batch', type=int, default=10)
parser.add_argument('--sample_method', type=str, default="random")
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
      Dense(128, activation='relu'),
      Dense(32, activation='relu'),
      Dense(self.action_dim)
      # Note: if use softmax as activation, you have to provide all the values of possible actions
      # as the target updating
    ])
    learning_rate = optimizers.schedules.ExponentialDecay(
        initial_learning_rate = args.lr, decay_steps = 20, decay_rate = .5, staircase=True)
    model.compile(loss='mse', optimizer=Adam(learning_rate))
    return model

  def predict(self, state):
    return self.model.predict(state)

  def get_action(self, state, deterministic=True):
    q_value = self.predict(state)[0]
    if not deterministic:
      self.epsilon *= args.eps_decay
      self.epsilon = max(self.epsilon, args.eps_min)
      if np.random.random() < self.epsilon:
        return random.randint(0, self.action_dim - 1)
    return np.argmax(q_value)

  def train(self, states, targets):
    self.updates += 1
    self.model.train_on_batch(states, targets)


class Agent:
  def __init__(self, env:Maze):
    self.env = env
    self.buffer = ReplayBuffer(args.batch_size*3)
    self.state_dim = self.env.state_dim()
    self.action_dim = self.env.action_dim()

    self.model = QualityModel(self.state_dim, self.action_dim)
    self.target_model = QualityModel(self.state_dim, self.action_dim)
    self.target_update()
    self.mse_error = deque(maxlen=50)

  def target_update(self):
    weights = self.model.model.get_weights()
    self.target_model.model.set_weights(weights)

  def vec_reward(self, a, value):
    reward = np.ones((args.batch_size, self.action_dim)) * Maze.NEU_REWARD
    row, col = list(range(args.batch_size)), a
    reward[row, col] = value
    reward = np.exp(reward)
    reward = reward/np.sum(reward, axis=1, keepdims=True)
    return reward

  def replay(self):
    for _ in range(args.replay_batch):
      if self.buffer.size() < args.batch_size:
        break
      # Note: for end states, should use different update schema
      states, actions, rewards, next_states, end = self.buffer.sample()
      # rewards_vec = self.vec_reward(actions, rewards)
      row, col = list(range(args.batch_size)), actions
      targets = self.model.predict(states)
      next_q_values = self.target_model.predict(next_states)
      targets[row, col] = rewards + (1.0 - end)*next_q_values[row, col] * args.gamma
      self.model.train(states, targets)
      q = self.model.predict(states)
      self.mse_error.append(np.sqrt(np.mean((q - targets)**2)))

  def train(self):
    if args.train_policy == "on-policy":
      self.train_onpolicy()
    elif args.train_policy == "off-policy":
      self.train_offpolicy()

  def train_offpolicy(self):
    '''
    :return:
    Tends to fall into dead loop, eg (left -> right -> left -> right ....)
    '''
    # Note: pay attention in collecting states
    self.env.reset()
    for loc, state in self.env.iter_states():
      for action in range(self.action_dim):
        next_state, reward, done = self.env.action(action)
        self.buffer.put(state, action, reward, next_state, done or reward == Maze.FAIL_REWARD)
        self.env.place_rob(np.array(loc))
    for ep in range(args.episode):
      for _ in range(10):
        self.replay()
      # update target_network every 10 batch
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
      mse_error = np.mean([x for x in self.mse_error])
      print(f'Episode {ep}, action rewards:{total_reward}, mse_error = {mse_error}')


  def train_onpolicy(self):
    for ep in range(args.episode):
      updates = 0
      done = False
      state = self.env.reset()
      state_set = set([",".join(str(x) for x in state.reshape(-1))])
      step = 0
      while not done and step < 3*args.batch_size:
        action = self.model.get_action(state, False)
        action_set = set([action])
        loc = self.env.get_rob()
        next_state, reward, done = self.env.action(action)
        state_repr = ",".join(str(x) for x in next_state.reshape(-1))

        while reward == Maze.FAIL_REWARD or state_repr in state_set:
          self.env.place_rob(loc)
          self.buffer.put(state, action, reward, next_state, done or reward == Maze.FAIL_REWARD)
          if len(action_set) >= self.action_dim:
            done = True
            print(f"dead loop, force break!")
            break
          for a in range(self.action_dim):
            if a in action_set:
              continue
            action_set.add(a)
            action = a
            break
          next_state, reward, done = self.env.action(action)
          state_repr = ",".join(str(x) for x in next_state.reshape(-1))
        self.buffer.put(state, action, reward, next_state, done or reward == Maze.FAIL_REWARD)
        state_set.add(state_repr)
        state = next_state
        step += 1

      if self.buffer.size() >= args.batch_size:
        self.replay()
        self.target_update()
        updates += 1

      if updates > 0:
        state = self.env.reset()
        print(f"iter whole locations ...")
        for rob, s in self.env.iter_states():
          act = self.model.get_action(s)
          print(f'      at {rob}, action = {act}')
        state = self.env.reset()
        step = 0
        reward = Maze.NEU_REWARD
        while reward != Maze.FAIL_REWARD and reward != Maze.WIN_REWARD and step < 10:
          self.env.print()
          action = self.model.get_action(state)
          print(f"action: {action}")
          next_state, reward, done = self.env.action(action)
          state = next_state
          step += 1
        mse_error = np.mean([x for x in self.mse_error])
        print(f'Episode {ep}, mse_error = {mse_error}, updates = {updates}')
      else:
        print(f'Episode {ep}, updates = {updates}')


def main():
  print(f"args = {args}")
  env = Maze(5, args.rand_seed)
  agent = Agent(env)
  agent.train()


if __name__ == "__main__":
  main()
