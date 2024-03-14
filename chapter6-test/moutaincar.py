import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v2 as tf
import time
from tensorflow import keras
import tensorflow as ttf

env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('位置范围 = {}'.format((env.unwrapped.min_position, env.unwrapped.max_position)))
print('速度范围 = {}'.format((-env.unwrapped.max_speed, env.unwrapped.max_speed)))
print('目标位置 = {}'.format(env.unwrapped.goal_position))

print(ttf.__version__)

if ttf.executing_eagerly():  
    print("Eager Execution is enabled.")  
else:  
    print("Eager Execution is disabled. TensorFlow is running in Graph Execution mode.")

class Chart:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)
    def plot(self, episode_rewards):
        self.ax.clear()
        self.ax.plot(episode_rewards)
        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('episode reward')
        self.fig.canvas.draw()

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['observation', 'action', 'reward',
                'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = np.asarray(args, dtype=object)
        self.count = self.count + 1
        self.i = self.count % self.capacity

    def sample(self, size):
        indices = np.random.choice(min(self.count, self.capacity), size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.01,
             replayer_capacity=10000, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity) # 经验回放

        self.evaluate_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, **net_kwargs) # 评估网络
        self.target_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, **net_kwargs) # 目标网络

        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, input_size, hidden_sizes, output_size,
                activation=tf.nn.relu, output_activation=None,
                learning_rate=0.01):

        model = keras.Sequential() # 构建网络

        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, **kwargs))

        model.add(keras.layers.Dense(units=output_size,
                activation=output_activation)) # 输出层
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                done) # 存储经验

        observations, actions, rewards, next_observations, terminateds = \
                self.replayer.sample(self.batch_size) # 经验回放

        next_qs = self.target_net(next_observations, training=False)
        next_max_qs = next_qs.numpy().max(axis=-1)
    
        us = rewards + self.gamma * (1. - terminateds) * next_max_qs

        targets = self.evaluate_net(observations, training=False)  
        targets_numpy = targets.numpy()  
        targets_numpy[np.arange(us.shape[0]), actions] = us  
        targets = tf.convert_to_tensor(targets_numpy, dtype=tf.float32)

        self.evaluate_net.fit(observations, targets, verbose=0)

        if done: # 更新目标网络
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation): # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)

        observations = observation[np.newaxis] # 将observation扩展为一个batch，因为模型通常期望输入是batch格式的
        qs = self.evaluate_net(observations) # 使用模型进行推断
        qs_numpy = qs.numpy()  # 获取Q值的numpy数组 
        return np.argmax(qs_numpy) # 返回具有最大Q值的动作的索引

def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    real_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)

        real_reward = reward
        if next_observation[0] > -0.5 and next_observation[0] < 0.5:
            reward = 10 * (next_observation[0] + 0.5)
        elif next_observation[0] >= 0.5:
            reward = 10000
        elif next_observation[0] <= -0.5:
            reward = -0.1

        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation,
                    done)
        if done:
            break
        observation = next_observation
    print('episode_reward:{}, count:{}'.format(episode_reward, agent.replayer.count))
    return episode_reward


net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.01}
agent = DQNAgent(env, net_kwargs=net_kwargs)

# 训练
episodes = 500
episode_rewards = []
chart = Chart()
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True, render=True)
    episode_rewards.append(episode_reward)
#     chart.plot(episode_rewards)
# chart.fig.show()

# 测试
agent.epsilon = 0. # 取消探索
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))