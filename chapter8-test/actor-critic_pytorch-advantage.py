import numpy as np
import matplotlib.pyplot as plt
import gym
import torch  
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F
import random

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
        self.observation_n = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma

        self.actor_net = self._build_network(hidden_sizes=actor_kwargs.get('hidden_sizes'),
                                             input_size=self.observation_n,
                                             output_size=self.action_n,
                                             output_activation=nn.Softmax)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(),
                                          lr=actor_kwargs.get('learning_rate', 0.001))
        
        # Critic网络：输出状态价值V(s)
        self.critic_net = self._build_network(hidden_sizes=critic_kwargs.get('hidden_sizes'),
                                              input_size=self.observation_n,
                                              output_size=1)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(),
                                          lr=critic_kwargs.get('learning_rate', 0.001))

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

    def decide(self, observation):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor_net(obs_tensor)
            dist = torch.distributions.Categorical(action_probs)
            return dist.sample().item()

    def learn(self, observation, action, reward, next_observation,
            terminated):
        # 转换数据为Tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        next_obs_tensor = torch.FloatTensor(next_observation).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        action_tensor = torch.LongTensor([action])

        # 训练Critic网络
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            if terminated:
                target_u = reward_tensor
            else:
                next_u = self.critic_net(next_obs_tensor)
                target_u = reward_tensor + self.gamma * next_u

        current_u = self.critic_net(obs_tensor)

        td_error = target_u - current_u

        critic_loss = nn.MSELoss()(current_u, target_u)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 训练Actor网络
        self.actor_optimizer.zero_grad()
        action_probs = self.actor_net(obs_tensor)
        log_prob = torch.log(action_probs[0, action] + 1e-10)

        actor_loss = -(target_u - current_u).detach() * log_prob
        actor_loss.backward()
        self.actor_optimizer.step()

def play_actor_critic(env, agent, train=False, render=False):
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
            agent.learn(observation, action, reward, next_observation, terminated)
        if terminated:
            break
        step += 1
        observation = next_observation
    return episode_reward

env = gym.make('Acrobot-v1')
env.seed(0)  # 设置环境随机种子

actor_kwargs = {'hidden_sizes' : [64, 32], 'learning_rate' : 0.001}
critic_kwargs = {'hidden_sizes' : [64, 32], 'learning_rate' : 0.001}
agent = QActorCriticAgent(env, actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs)

# 训练
episodes = 1500
episode_rewards = []
chart = Chart()
for episode in range(episodes):
    episode_reward = play_actor_critic(env, agent, train=True, render=False)
    episode_rewards.append(episode_reward)
    print(f"Episode {episode} reward: {episode_reward}")
    chart.plot(episode_rewards)

# 测试
episode_rewards = [play_actor_critic(env, agent, render=True) for _ in range(50)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))