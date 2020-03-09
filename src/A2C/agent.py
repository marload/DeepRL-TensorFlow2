import tensorflow as tf

import numpy as np

from actor import Actor
from critic import Critic


class Agent:
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        ACTOR_LEARNING_RATE = 0.0001
        CRITIC_LEARNING_RATE = 0.001

        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_boundm, self.std_bound, ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, CRITIC_LEARNING_RATE)
