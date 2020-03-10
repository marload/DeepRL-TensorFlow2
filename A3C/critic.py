import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

import numpy as np


class Critic:
    def __init__(self, state_dim, learning_rate=None):
        self.state_dim = state_dim

        if not learning_rate == None:
            self.loss = tf.keras.losses.MeanSquaredError()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model = self.create_model()

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(128, activation='relu')(state_input)
        dense_2 = Dense(128, activation='relu')(dense_1)
        output = Dense(1, activation='linear')(dense_2)
        return tf.keras.models.Model(state_input, output)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_values = self.model(states)
            loss = self.loss(v_values, td_targets)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
