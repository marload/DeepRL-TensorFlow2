import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

import numpy as np


class Critic:
    def __init__(self, state_dim, learning_rate):
        self.state_dim = state_dim

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model = self.create_model()

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1, activation='linear')
        ])

    def predict(self, states):
        return self.model.predict(states)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_values = self.predict(states)
            loss = self.loss(v_values, td_targets)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
