#!/usr/bin/env python3
# -*-coding: utf-8 -*-
import numpy as np
import threading
import tensorflow as tf
from queue import Queue
from tensorflow import keras
from tensorflow.keras.layers import Dense
import gym
from collections import deque

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.InteractiveSession(config=config)


class Actor(keras.Model):
    def __init__(self, action_size):
        super().__init__()
        self.hidden1 = Dense(16, activation='relu')
        self.hidden2 = Dense(16, activation='relu')
        self.hidden3 = Dense(16, activation='relu')
        self.policy = Dense(action_size, activation='softmax')

    @tf.function
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.hidden3(x)
        probs = self.policy(x)
        return probs


class Critic(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = Dense(16, activation='relu')
        self.hidden2 = Dense(16, activation='relu')
        self.hidden3 = Dense(16, activation='relu')
        self.values = Dense(1, activation='linear')

    @tf.function
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.values(x)
        return x


class A3C:
    def __init__(self, game_name, log_dir='Test'):
        self.game_name = game_name
        env = gym.make(self.game_name)
        self.global_actor = Actor(env.action_space.n)
        self.global_critic = Critic()
        self.global_actor(tf.random.uniform((1, env.observation_space.shape[0])))
        self.global_critic(tf.random.uniform((1, env.observation_space.shape[0])))

        self.log_dir = log_dir

    def train(self, actor_opt, critic_opt, update_freq=20, thread_num=1, gamma=0.95, entro=0.0001):
        res_queue = Queue()

        workers = [Agent(self.game_name, self.global_actor, self.global_critic, actor_opt, critic_opt,
                         update_freq, res_queue, i, gamma=gamma, entro=entro)
                   for i in range(thread_num)]

        for i, worker in enumerate(workers):
            print(f'Starting worker {i}')
            worker.start()

        [w.join() for w in workers]

        return None

    def evaluate(self):
        state = np.array(self.env.states)
        probs = self.global_model(tf.constant(value=state, dtype=tf.float32))[0]
        action = tf.argmax(probs, axis=1) + 1
        return action


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Agent(threading.Thread):
    global_episode = 0
    save_lock = threading.Lock()

    def __init__(self,
                 game_name,
                 global_actor,
                 global_critic,
                 actor_opt,
                 critic_opt,
                 update_freq,
                 result_queue,
                 idx,
                 gamma=0.95,
                 entro=0.0001,
                 goal=190):

        super(Agent, self).__init__()

        self.update_freq = update_freq
        self.gamma = gamma
        self.entro = entro

        self.env = gym.make(game_name)
        self.result_queue = result_queue
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor_opt = tf.optimizers.Adam(0.0025)
        self.critic_opt = tf.optimizers.Adam(0.005)
        self.local_actor = Actor(self.env.action_space.n)
        self.local_critic = Critic()
        self.agent_idx = idx
        self.goal = goal

    def choose_action(self, state):
        probs = self.local_actor(tf.constant([state], dtype=tf.float32))
        action = np.random.choice(self.env.action_space.n, p=probs.numpy()[0])
        return action

    def run(self):
        mem = Memory()
        mean_reward = deque(maxlen=10)
        mean_reward.append(0)

        while np.mean(mean_reward) < self.goal:
            state = self.env.reset()
            reward = 0
            step = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, r, done, _ = self.env.step(action)
                mem.store(state, action, r)
                reward += r
                step += 1
                if step == self.update_freq or done:
                    self.learn(mem, next_state, done)
                    mem.clear()
                    step = 0
                state = next_state
            Agent.global_episode += 1
            mean_reward.append(reward)
            print(f'Agent: {self.agent_idx} | episodes: {Agent.global_episode} '
                  f'| reward: {reward} | mean: {np.mean(mean_reward)}')

        self.result_queue.put(None)

    def learn(self, mem, next_state, done):
        if done:
            reward_sum = 0
        else:
            reward_sum = self.local_critic(tf.constant([next_state], dtype=tf.float32)).numpy()[0, 0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in mem.rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = tf.constant(discounted_rewards, dtype=tf.float32)[:, None]
        states = tf.constant(mem.states, dtype=tf.float32)
        actions = tf.constant(mem.actions, dtype=tf.int32)

        print(states.shape, actions.shape, discounted_rewards.shape)
        self._update(states, actions, discounted_rewards)

        self.local_actor.set_weights(self.global_actor.get_weights())
        self.local_critic.set_weights(self.global_critic.get_weights())
        return None

    @tf.function
    def _update(self, states, actions, discounted_rewards):
        """update critic"""
        with tf.GradientTape() as tape:
            discounted_rewards = tf.stop_gradient(discounted_rewards)

            state_value = self.local_critic(states)

            advantage = discounted_rewards - state_value

            critic_loss = tf.reduce_mean(advantage ** 2)

        critic_grad = tape.gradient(critic_loss, self.local_critic.trainable_weights)

        self.critic_opt.apply_gradients(zip(critic_grad, self.global_critic.trainable_weights))

        """update actor"""
        with tf.GradientTape() as tape:
            probs = self.local_actor(states)

            advantage = tf.stop_gradient(tf.squeeze(advantage))
            entropy = -tf.reduce_sum(probs * tf.math.log(probs), axis=-1)

            actions_encode = tf.one_hot(actions, self.env.action_space.n, 1.0, 0.0)
            actions_prob = tf.reduce_sum(probs * actions_encode, axis=-1)

            actor_loss = tf.reduce_mean(-tf.math.log(actions_prob) * advantage - self.entro * entropy)

        actor_grad = tape.gradient(actor_loss, self.local_actor.trainable_weights)

        self.actor_opt.apply_gradients(zip(actor_grad, self.global_actor.trainable_weights))


if __name__ == '__main__':
    game_name = 'CartPole-v0'
    actor_opt = tf.optimizers.Adam(0.0025)
    critic_opt = tf.optimizers.Adam(0.005)
    agent = A3C(game_name)
    agent.train(actor_opt, critic_opt, update_freq=20, thread_num=3, gamma=0.95)
