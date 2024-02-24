import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v2 as tf
import time
from tensorflow import keras

env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('位置范围 = {}'.format((env.unwrapped.min_position, env.unwrapped.max_position)))
print('速度范围 = {}'.format((-env.unwrapped.max_speed, env.unwrapped.max_speed)))
print('目标位置 = {}'.format(env.unwrapped.goal_position))

# positions, velocities = [], []
# observation = env.reset()
# action = 2
# while True:
#     env.render()
#     # time.sleep(1)
#     positions.append(observation[0])
#     velocities.append(observation[1])
#     next_observation, reward, done, _ = env.step(action)
#     #human's policy
#     if action == 2:
#         if next_observation[1] <= 0.0:
#             action = 0
#     elif action == 0:
#         if next_observation[1] >= 0.0:
#             action = 2
#     if done:
#         break
#     observation = next_observation
# if next_observation[0] > 0.5:
#     print('成功达到')
# else:
#     print('失败退出')
# env.close()
# fig, ax = plt.subplots()
# ax.plot(positions, label='position')
# ax.plot(velocities, label='velocity')
# ax.legend()
# plt.show()
# print('Needed Steps {}', len(positions))

class TileCoder:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}
    def get_feature(self, codeword): # [codeword] ==> [feature]
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:
            return hash(codeword) % self.features
        self.codebook[codeword] = count
        return count
    def __call__(self, floats=(), ints=()): # [observation, action] ==> [codewords] ==> [features]
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer) / self.layers) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features

class SARSAAgent:
    def __init__(self, env, layers=8, features=1893, gamma=1., learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n # 动作数
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low # 观测空间范围
        self.encoder = TileCoder(layers, features) # 砖瓦编码器
        self.w = np.zeros(features) # 权重
        self.gamma = gamma # 折扣
        self.learning_rate = learning_rate # 学习率
        self.epsilon = epsilon # 探索

    def encode(self, observation, action): # 编码 [observation, action] ==> [code] ==> [features]
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.encoder(states, actions)

    def get_q(self, observation, action): # 动作价值  [observation, action] ==> [code] ==> [features] ==> [Q(w)]
        features = self.encode(observation, action)
        return self.w[features].sum()

    def decide(self, observation): # 判决
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [self.get_q(observation, action) for action in range(self.action_n)]
            return np.argmax(qs)

    def learn(self, observation, action, reward, next_observation, done, next_action):
        u = reward + (1. - done) * self.gamma * self.get_q(next_observation, action)
        td_error = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += (self.learning_rate * td_error)

class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env, layers=8, features=1893, gamma=1., learning_rate=0.03, epsilon=0.001, lambd=0.9):
        super().__init__(env=env, layers=layers, features=features, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lambd = lambd
        self.z = np.zeros(features)
    def learn(self, observation, action, reward, next_observation, done, next_action):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(next_observation, next_action))
            self.z *= (self.gamma * self.lambd)
            features = self.encode(observation, action)
            self.z[features] = 1.
        td_error = u - self.get_q(observation, action)
        self.w += (self.learning_rate * td_error * self.z)
        if done:
            self.z = np.zeros_like(self.z)

def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)
        if train:
            agent.learn(observation, action, reward, next_observation, done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    print('Needed Steps:{}',episode_reward)
    return episode_reward

agent = SARSALambdaAgent(env)

# 训练
episodes = 180
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()

# 测试
agent.epsilon = 0.
episode_rewards = [play_sarsa(env, agent, render=True) for _ in range(10)]
env.close()
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

