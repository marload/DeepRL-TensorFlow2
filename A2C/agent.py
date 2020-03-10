import tensorflow as tf

import numpy as np

from actor import Actor
from critic import Critic


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.GAMMA = 0.95
        self.BATCH_SIZE = 32

        ACTOR_LEARNING_RATE = 0.0001
        CRITIC_LEARNING_RATE = 0.001

        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound, ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, CRITIC_LEARNING_RATE)

    def td_target(self, reward, next_v_value, done):
        reward = reward * 0.01
        if done:
            return reward
        else:
            return reward + self.GAMMA * next_v_value

    def advantage(self, td_target, baselines):
        return td_target - baselines

    def unpack_batch(self, batch):
        unpacked = batch[0]
        for elem in batch[1:]:
            unpacked = np.append(unpacked, elem, axis=0)
        return unpacked

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []
            time_stamp, total_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                action = self.actor.get_action(state)

                action = np.clip(action, -self.action_bound, self.action_bound)
                next_state, reward, done, _ = self.env.step(action)
                reward = float(reward)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                v_value = self.critic.model(state)
                next_v_value = self.critic.model(next_state)
                td_target = self.td_target(reward, next_v_value, done)
                advantage = self.advantage(td_target, v_value)

                batch_state.append(state)
                batch_action.append(action)
                batch_td_target.append(td_target)
                batch_advantage.append(advantage)

                if len(batch_state) >= self.BATCH_SIZE or done:
                    states = self.unpack_batch(batch_state)
                    actions = self.unpack_batch(batch_action)
                    td_targets = self.unpack_batch(batch_td_target)
                    advantages = self.unpack_batch(batch_advantage)

                    batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

                    self.actor.train(states, actions, advantages)
                    self.critic.train(states, td_targets)

                time_stamp += 1
                state = next_state[0]
                total_reward += reward[0][0]

            print('Episode: [{}], TimeStep: [{}], Reward: [{}]'.format(
                ep, time_stamp, total_reward))
