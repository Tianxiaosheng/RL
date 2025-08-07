import numpy as np
np.random.seed(0)
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v2 as tf
tf.random.set_seed(0)
from tensorflow import keras

class Chart:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)

    def plot(self, episode_rewards):
        self.ax.clear()
        self.ax.plot(episode_rewards)
        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('episode reward')
        self.ax.set_title('Episode Rewards')
        self.fig.canvas.draw()

class QActorCriticAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1.

        self.actor_net = self.build_network(output_size=self.action_n,
                output_activation=tf.nn.softmax,
                loss=tf.losses.categorical_crossentropy,
                **actor_kwargs)
        self.critic_net = self.build_network(output_size=self.action_n,
                **critic_kwargs)

    def build_network(self, hidden_sizes, output_size, input_size=None,
                activation=tf.nn.relu, output_activation=None,
                loss=tf.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)
            model.add(keras.layers.Dense(units=hidden_size,
                    activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis], verbose=0)[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, next_observation,
            terminated, next_action=None):
        # 训练执行者网络
        x = observation[np.newaxis]
        u = self.critic_net.predict(x, verbose=0)
        q = u[0, action]
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor,
                    1e-6, 1.))
            loss_tensor = -self.discount * q * logpi_tensor
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(
                grad_tensors, self.actor_net.variables))

        # 训练评论者网络
        u[0, action] = reward
        if not (terminated):
            q = self.critic_net.predict(
                    next_observation[np.newaxis], verbose=0)[0, next_action]
            u[0, action] += self.gamma * q
        self.critic_net.fit(x, u, verbose=0)

        if terminated:
            self.discount = 1.
        else:
            self.discount *= self.gamma


def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, terminated, _ = env.step(action)
        episode_reward += reward
        if terminated:
            if train:
                agent.learn(observation, action, reward,
                    next_observation, terminated)
            break
        next_action = agent.decide(next_observation)
        if train:
            agent.learn(observation, action, reward, next_observation,
                    terminated, next_action)
        observation, action = next_observation, next_action
    return episode_reward

env = gym.make('Acrobot-v1')

actor_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.0001}
critic_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.0002}
agent = QActorCriticAgent(env, actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs)

# 训练
episodes = 150
episode_rewards = []
chart = Chart()
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True, render=False)
    episode_rewards.append(episode_reward)
    print(f"Episode {episode} reward: {episode_reward}")
    chart.plot(episode_rewards)

# 测试
episode_rewards = [play_sarsa(env, agent, render=True) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))