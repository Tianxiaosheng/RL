import numpy as np
np.random.seed(0)
import gym
import pandas as pd
import matplotlib.pyplot as plt
import torch  
import torch.nn as nn  
import torch.optim as optim
import random

from collections import deque, namedtuple       # 队列类型

env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('位置范围 = {}'.format((env.unwrapped.min_position, env.unwrapped.max_position)))
print('速度范围 = {}'.format((-env.unwrapped.max_speed, env.unwrapped.max_speed)))
print('目标位置 = {}'.format(env.unwrapped.goal_position))

class Chart:
    def __init__(self, plt):
        self.plt = plt
    def plot(self, episode_rewards):
        self.plt(episode_rewards)
        self.plt.show()

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch_data)
        return state, action, reward, next_state, done
 
    def push(self, *args):
        # *args: 把传进来的所有参数都打包起来生成元组形式
        # self.push(1, 2, 3, 4, 5)
        # args = (1, 2, 3, 4, 5)
        self.memory.append(self.Transition(*args))
 
    def __len__(self):
        return len(self.memory)

# 定义DQN网络结构
class DQNNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):  
        super(DQNNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.ReLU()) for i in range(len(hidden_sizes)-1)],
            nn.Linear(hidden_sizes[-1], output_size)
        )

    def forward(self, x):
        return self.model(x)

# 定义DQNAgent类  
class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.01,  
                 replayer_capacity=20000, batch_size=64):
        self.observation_dim = env.observation_space.shape[0]  
        self.action_n = env.action_space.n  
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.batch_size = batch_size  
        self.capacity = replayer_capacity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.count = 0

        # 初始化评估网络和目标网络  
        self.evaluate_net = DQNNet(self.observation_dim, net_kwargs.get('hidden_sizes', [64, 64]), self.action_n).to(self.device)  
        self.target_net = DQNNet(self.observation_dim, net_kwargs.get('hidden_sizes', [64, 64]), self.action_n).to(self.device)

        # 初始化目标网络的权重与评估网络相同  
        self.update_target_net()

        # 定义优化器
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=net_kwargs.get('learning_rate', 0.01))

        # 定义损失函数  
        self.criterion = nn.MSELoss()  

    def update_target_net(self):  
        self.target_net.load_state_dict(self.evaluate_net.state_dict())

    def learn(self,transition_dict):
        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1) # 扩充维度
        rewards = np.expand_dims(transition_dict.reward, axis=-1) # 扩充维度
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1) # 扩充维度

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        predict_q_values = self.evaluate_net(states).gather(1, actions)

        # 计算目标Q值
        with torch.no_grad():
            # max(1) 即 max(dim=1)在行向找最大值，这样的话shape(64, ), 所以再加一个view(-1, 1)扩增至(64, 1)
            max_next_q_values = self.target_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        l = self.criterion(predict_q_values, q_targets)

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
 
        if transition_dict.done:
            # copy model parameters
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
 
        self.count += 1

    def decide(self, observation):
        # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)

        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        action = torch.argmax(self.evaluate_net(state)).item()

        return action

def play_qlearning(env, agent, repalymemory, train=False, render=False):
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
        reward = real_reward
        repalymemory.push(observation, action, reward, next_observation, done)
        episode_reward += reward

        if train:
            if len(repalymemory) > agent.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = repalymemory.sample(agent.batch_size)
                T_data = repalymemory.Transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                agent.learn(T_data)
        if done:
            break
        observation = next_observation
    print('episode_reward:{}, count:{}'.format(episode_reward, agent.count))
    return episode_reward

net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.001}
agent = DQNAgent(env, net_kwargs=net_kwargs)

# 训练
episodes = 500
episode_rewards = []

replaymemory = DQNReplayer(capacity=60000)
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, replaymemory, train=True, render=True)
    episode_rewards.append(episode_reward)
print('Training->平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

plt.plot(episode_rewards)
plt.show()

# 测试
agent.epsilon = 0. # 取消探索
episode_rewards = [play_qlearning(env, agent, replaymemory, render=True) for _ in range(100)]
print('Evaluate->平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))
