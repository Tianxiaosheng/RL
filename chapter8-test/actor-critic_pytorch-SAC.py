import numpy as np
import matplotlib.pyplot as plt
import gym
import torch  
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple       # 队列类型

# 设置随机种子
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

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

class SACAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99, 
                 alpha=0.2, net_learning_rate=0.1, replayer_capacity=10000, 
                 batch_size=64, target_update_freq=100):
        self.observation_n = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.net_learning_rate = net_learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 经验回放缓冲区
        self.replayer = DQNReplayer(replayer_capacity)

        # Actor网络：输出动作概率分布
        self.actor_net = self._build_network(hidden_sizes=actor_kwargs.get('hidden_sizes'),
                                             input_size=self.observation_n,
                                             output_size=self.action_n,
                                             output_activation=nn.Softmax)
        self.actor_net.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(),
                                          lr=actor_kwargs.get('learning_rate', 0.001))
        
        # 双Q网络
        self.q0_net = self._build_network(hidden_sizes=critic_kwargs.get('hidden_sizes'),
                                          input_size=self.observation_n,
                                          output_size=self.action_n)
        self.q0_net.to(self.device)
        self.q1_net = self._build_network(hidden_sizes=critic_kwargs.get('hidden_sizes'),
                                          input_size=self.observation_n,
                                          output_size=self.action_n)
        self.q1_net.to(self.device)
        self.q0_optimizer = optim.Adam(self.q0_net.parameters(),
                                       lr=critic_kwargs.get('learning_rate', 0.001))
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(),
                                       lr=critic_kwargs.get('learning_rate', 0.001))
        
        # V网络（评估网络和目标网络）
        self.v_evaluate_net = self._build_network(hidden_sizes=critic_kwargs.get('hidden_sizes'),
                                                  input_size=self.observation_n,
                                                  output_size=1)
        self.v_evaluate_net.to(self.device)
        self.v_target_net = self._build_network(hidden_sizes=critic_kwargs.get('hidden_sizes'),
                                                input_size=self.observation_n,
                                                output_size=1)
        self.v_target_net.to(self.device)
        self.v_optimizer = optim.Adam(self.v_evaluate_net.parameters(),
                                      lr=critic_kwargs.get('learning_rate', 0.001))
        
        # 初始化目标网络
        self.update_target_net()

    def _build_network(self, hidden_sizes, output_size, input_size=None,
                activation=nn.ReLU, output_activation=None):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        if output_activation is not None:
            if output_activation == nn.Softmax:
                layers.append(nn.Softmax(dim=1))
            else:
                layers.append(output_activation())
        return nn.Sequential(*layers)

    def update_target_net(self):
        """软更新目标网络"""
        for target_param, evaluate_param in zip(self.v_target_net.parameters(), 
                                               self.v_evaluate_net.parameters()):
            target_param.data.copy_(
                (1 - self.net_learning_rate) * target_param.data + 
                self.net_learning_rate * evaluate_param.data
            )

    def decide(self, observation):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor_net(obs_tensor)
            dist = torch.distributions.Categorical(action_probs)
            return dist.sample().item()

    def learn(self, transition_dict):
        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1) # 扩充维度
        rewards = np.expand_dims(transition_dict.reward, axis=-1) # 扩充维度
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1) # 扩充维度

        obs_tensor = torch.tensor(states, dtype=torch.float).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_obs_tensor = torch.tensor(next_states, dtype=torch.float).to(self.device)
        terminated_tensor = torch.tensor(dones, dtype=torch.float).to(self.device)

        print("obs_tensor: ", obs_tensor)
        print("action_tensor: ", action_tensor)
        print("reward_tensor: ", reward_tensor)
        print("next_obs_tensor: ", next_obs_tensor)
        print("terminated_tensor: ", terminated_tensor)

        # 获取当前策略和Q值
        pis = self.actor_net(obs_tensor)  # [batch_size, action_n]
        q0s = self.q0_net(obs_tensor)    # [batch_size, action_n]
        q1s = self.q1_net(obs_tensor)    # [batch_size, action_n]

        # 训练Actor网络（SAC损失）
        self.actor_optimizer.zero_grad()
        q01s = torch.min(q0s, q1s)  # 取两个Q值的最小值

        # SAC的Actor损失：最大化期望Q值 + 熵正则化
        log_pis = torch.log(torch.clamp(pis, 1e-10, 1.0))  # 数值稳定性
        entropy = -torch.sum(pis * log_pis, dim=1)  # 策略熵
        q_pi = torch.sum(pis * q01s, dim=1)  # 期望Q值
        actor_loss = torch.mean(torch.sum(pis * (self.alpha * log_pis - q01s), dim=1))
        actor_loss.backward()
        self.actor_optimizer.step()

        # 训练V网络
        self.v_optimizer.zero_grad()
        with torch.no_grad():
            # 计算目标V值：E[Q - α*log π]
            entropic_q01s = q01s - self.alpha * log_pis
            v_targets = torch.sum(pis * entropic_q01s, dim=1)
        
        v_pred = self.v_evaluate_net(obs_tensor).squeeze()
        v_loss = F.mse_loss(v_pred, v_targets)
        v_loss.backward()
        self.v_optimizer.step()

        # 训练Q网络
        self.q0_optimizer.zero_grad()
        self.q1_optimizer.zero_grad()
        
        with torch.no_grad():
            next_vs = self.v_target_net(next_obs_tensor).squeeze()
            q_targets = reward_tensor + self.gamma * (1 - terminated_tensor) * next_vs
        
        # 更新Q值
        q0_pred = self.q0_net(obs_tensor)
        q1_pred = self.q1_net(obs_tensor)

        # 只更新实际采取的动作的Q值
        q0_target = q0_pred.clone()
        q1_target = q1_pred.clone()
        q0_target[range(self.batch_size), action_tensor] = q_targets
        q1_target[range(self.batch_size), action_tensor] = q_targets
        
        q0_loss = F.mse_loss(q0_pred, q0_target)
        q1_loss = F.mse_loss(q1_pred, q1_target)
        
        q0_loss.backward()
        q1_loss.backward()
        self.q0_optimizer.step()
        self.q1_optimizer.step()

        # 定期更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_net()

def play_sac(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    step = 0
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, terminated, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.replayer.push(observation, action, reward, next_observation, terminated)
            if len(agent.replayer) > agent.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch =\
                        agent.replayer.sample(agent.batch_size)
                T_data = agent.replayer.Transition(
                        state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                agent.learn(T_data)
        if terminated:
            break
        step += 1
        observation = next_observation
    return episode_reward

env = gym.make('Acrobot-v1')
env.seed(0)  # 设置环境随机种子

actor_kwargs = {'hidden_sizes' : [64, 32], 'learning_rate' : 0.0003}
critic_kwargs = {'hidden_sizes' : [64, 32], 'learning_rate' : 0.0003}
agent = SACAgent(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs,
                 alpha=0.1, batch_size=32, target_update_freq=50)

# 训练
episodes = 1000  # 增加训练轮数
episode_rewards = []
chart = Chart()

for episode in range(episodes):
    episode_reward = play_sac(env, agent, train=True, render=False)
    episode_rewards.append(episode_reward)
    print(f"Episode {episode} reward: {episode_reward}")
    chart.plot(episode_rewards)

# 测试
episode_rewards = [play_sac(env, agent, render=True) for _ in range(50)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))