import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import numpy as np


class Actor:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_bound,
        std_bound,
        learning_rate
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound

        self.optimizer = tf.keras.optimizers(learning_rate)
        self.model = self.create_model()

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(128, activation='relu')(state_input)
        dense_2 = Dense(128, activation='relu')(dense_1)
        mu_output = Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)
        std_output = Dense(self.action_dim, activatio='softplus')(dense_2)
        output = [mu_output, std_output]
        return tf.keras.models.Model(state_input, output)

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        mu, std = self.predict(state)
        mu, std = mu[0], std[0]
        std = np.clip(std, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu, std, size=self.action_dim)
        return action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(mu, std, actions)
        return -tf.reduce_sum(log_policy_pdf * advantages)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.predict(states)
            loss = self.loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
