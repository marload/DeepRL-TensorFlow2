import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

from Env import Maze
import argparse
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--rand_seed', type=int, default=None)
parser.add_argument('--episode', type=int, default=32)

args = parser.parse_args()
print(f"args = {args}")

CUR_EPISODE = 0


class ActorCritic:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.act_opt = keras.optimizers.Adam(args.actor_lr)
        self.cri_opt = keras.optimizers.Adam(args.critic_lr)
        self.entropy_beta = 0.01
        self.act_model, self.cri_model = self.create_model()

    def create_model(self):
        inp = Input((self.state_dim,))
        share_model = Dense(16, activation='relu')(
                Dense(32, activation='relu')(inp))
        act_out = Dense(self.action_dim, activation='softmax')(share_model)
        outp = Dense(16, activation='relu')(share_model)
        cri_out = Dense(1, activation="linear")(outp)
        return keras.Model(inp, act_out), keras.Model(inp, cri_out)

    def compute_act_loss(self, actions, logits, advantages):
        ce_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        entropy_loss = keras.losses.CategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        # Q: is it correct to use this loss to get gradient ? yes!
        # Q: sum or sub of the two ?
        return policy_loss + self.entropy_beta * entropy
        # return policy_loss - self.entropy_beta * entropy

    def train_actor(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.act_model(states, training=True)
            loss = self.compute_act_loss(
                actions, logits, advantages)
        grads = tape.gradient(loss, self.act_model.trainable_variables)
        self.act_opt.apply_gradients(zip(grads, self.act_model.trainable_variables))
        return loss

    def compute_cri_loss(self, v_pred, td_targets):
        mse = keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train_critic(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.cri_model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_cri_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.cri_model.trainable_variables)
        self.cri_opt.apply_gradients(zip(grads, self.cri_model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env_name):
        env = Maze(5, rs=args.rand_seed)
        self.env_name = env_name
        self.state_dim = env.state_dim()
        self.action_dim = env.action_dim()

        self.global_a3c = ActorCritic(self.state_dim, self.action_dim)
        self.num_workers = cpu_count()

    def train(self):
        max_episodes = args.episode
        workers = []

        for i in range(self.num_workers):
            env = Maze(5, rs=args.rand_seed)
            workers.append(WorkerAgent(env, self.global_a3c, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class WorkerAgent(Thread):
    def __init__(self, env, global_a3c, max_episodes):
        Thread.__init__(self)
        self.lock = Lock()
        self.env = env
        self.state_dim = self.env.state_dim()
        self.action_dim = self.env.action_dim()

        self.max_episodes = max_episodes
        self.global_a3c = global_a3c
        a3c_local = ActorCritic(self.state_dim, self.action_dim)
        self.actor = a3c_local.act_model
        self.critic = a3c_local.cri_model

        self.actor.set_weights(self.global_a3c.act_model.get_weights())
        self.critic.set_weights(self.global_a3c.cri_model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            td_targets[k] = cumulative = args.gamma * cumulative + rewards[k]
        return td_targets

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        global CUR_EPISODE

        while self.max_episodes >= CUR_EPISODE:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                # self.env.render()
                probs = self.actor.predict(
                    np.reshape(state, [1, self.state_dim]))
                # Note: not select the max!
                action = np.random.choice(self.action_dim, p=probs[0])

                next_state, reward, done = self.env.action(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) >= args.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)

                    next_v_value = self.critic.model.predict(next_state)
                    td_targets = self.n_step_td_target(
                        rewards, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states)
                    
                    with self.lock:
                        self.global_a3c.train_actor(
                            states, actions, advantages)
                        self.global_a3c.train_critic(
                            states, td_targets)

                        self.actor.set_weights(
                            self.global_a3c.act_model.get_weights())
                        self.critic.model.set_weights(
                            self.global_a3c.cri_model.get_weights())
                        # show temp results:
                        for rob, s in self.env.iter_states():
                            probs = self.actor.predict(s)
                            print(f'      at {rob}, action probs = {probs}')
                        done, total_reward = False, 0
                        state = self.env.reset()
                        step = 0
                        while not done and step < 10:
                            self.env.print()
                            action = np.argmax(self.actor.predict(state))
                            print(f"action: {action}")
                            next_state, reward, done = self.env.action(action)
                            total_reward += reward
                            state = next_state
                            step += 1
                        print(f'Episode {CUR_EPISODE}, rewards:{total_reward}')

                    state_batch = []
                    action_batch = []
                    reward_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(CUR_EPISODE, episode_reward))
            CUR_EPISODE += 1

    def run(self):
        self.train()


def main():
    env_name = 'Maze'
    agent = Agent(env_name)
    agent.train()


if __name__ == "__main__":
    main()
