import tensorflow as tf

import numpy as np

from threading import Thread
from multiprocessing import cpu_count
import gym

from actor import Actor
from critic import Critic

GLOBAL_EPISODE = 0


class GlobalAgent:
    def __init__(self, env_name):
        self.env_name = env_name

        self.NUM_WORKERS = cpu_count()

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]
        std_bound = [1e-2, 1.0]

        ACTOR_LEARNING_RATE = 0.0001
        CRITIC_LEARNING_RATE = 0.001
        ENTROPY_BETA = 0.01

        self.global_actor = Actor(
            state_dim, action_dim, action_bound, std_bound, ACTOR_LEARNING_RATE, ENTROPY_BETA)
        self.global_critic = Critic(state_dim, CRITIC_LEARNING_RATE)

    def train(self, max_episode=1000):
        max_episode = max_episode
        workers = []
        for i in range(self.NUM_WORKERS):
            env = gym.make(self.env_name)
            worker_name = 'Worker{}'.format(i+1)
            workers.append(WorkerAgent(
                env, worker_name, self.global_actor, self.global_critic, max_episode))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class WorkerAgent(Thread):
    def __init__(
        self,
        env,
        worker_name,
        global_actor,
        global_critic,
        max_episode
    ):
        Thread.__init__(self)
        self.env = env
        self.worker_name = worker_name

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.GAMMA = 0.95
        self.T_MAX = 4
        self.max_episode = max_episode

        self.global_actor = global_actor
        self.global_critic = global_critic

        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    def unpack_batch(self, batch):
        unpacked = batch[0]
        for elem in unpacked[1:]:
            np.append(unpack_batch, elem, axis=0)
        return unpacked

    def t_step_td_target(self, rewards, next_v_value, done):
        rewards *= 0.01
        td_targets = np.zeros_like(rewards)
        accumulate = 0
        if not done:
            accumulate = next_v_value
        for idx in reversed(range(len(rewards))):
            accumulate = self.GAMMA * accumulate + rewards[idx]
            td_targets[idx] = accumulate
        return td_targets

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def run(self):
        global GLOBAL_EPISODE

        while self.max_episode >= GLOBAL_EPISODE:
            state = self.env.reset()
            time_stamp, total_reward, done = 0, 0, False
            batch_state, batch_action, batch_reward = [], [], []

            while not done:

                action = self.actor.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)
                next_state, reward, done, _ = self.env.step(action)
                reward = float(reward)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(reward)

                if len(batch_state) >= self.T_MAX or done:
                    states = self.unpack_batch(batch_state)
                    actions = self.unpack_batch(batch_action)
                    rewards = self.unpack_batch(batch_reward)
                    v_values = self.critic.model(states)
                    next_v_value = self.critic.model(next_state)

                    batch_state, batch_action, batch_reward = [], [], []

                    td_targets = self.t_step_td_target(
                        rewards, next_v_value, done)
                    advantages = self.advantage(td_targets, v_values)

                    self.global_actor.train(states, actions, advantages)
                    self.global_critic.train(states, td_targets)

                    self.actor.model.set_weights(
                        self.global_actor.model.get_weights())
                    self.critic.model.set_weights(
                        self.global_critic.model.get_weights())

                total_reward += reward[0][0]
                state = next_state[0]
                time_stamp += 1

            print('{} - EPISODE: [{}], TIMESTAMP: [{}], REWARD: [{}]'.format(
                self.worker_name, GLOBAL_EPISODE, time_stamp, total_reward))
            GLOBAL_EPISODE += 1
