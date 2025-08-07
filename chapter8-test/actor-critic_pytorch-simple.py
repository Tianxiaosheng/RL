import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import gym
import torch  
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F

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
        self.discount = 1.

        self.actor_net = self._build_network(hidden_sizes=actor_kwargs.get('hidden_sizes'),
                                             input_size=self.observation_n,
                                             output_size=self.action_n,
                                             output_activation=nn.Softmax)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(),
                                          lr=actor_kwargs.get('learning_rate', 0.001))
        
        # Critic网络：输出状态价值V(s)
        self.critic_net = self._build_network(hidden_sizes=critic_kwargs.get('hidden_sizes'),
                                              input_size=self.observation_n,
                                              output_size=self.action_n)
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
            layers.append(output_activation())
        return nn.Sequential(*layers)

    def decide(self, observation):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor_net(obs_tensor)
            dist = torch.distributions.Categorical(action_probs)
            return dist.sample().item()

    def learn(self, observation, action, reward, next_observation,
            terminated, next_action=None):
        # 转换数据为Tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        next_obs_tensor = torch.FloatTensor(next_observation).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        action_tensor = torch.LongTensor([action])

        # 训练Critic网络
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            if terminated:
                target_q = reward_tensor
            else:
                next_q = self.critic_net(next_obs_tensor)[0, next_action]
                target_q = reward_tensor + self.gamma * next_q

        current_q = self.critic_net(obs_tensor)[0, action].unsqueeze(0)

        cirtic_loss = nn.MSELoss()(current_q, target_q)
        cirtic_loss.backward()
        self.critic_optimizer.step()

        # 训练Actor网络
        self.actor_optimizer.zero_grad()
        action_probs = self.actor_net(obs_tensor)
        log_prob = torch.log(action_probs[0, action] + 1e-8)

        with torch.no_grad():
            q_value = self.discount * self.critic_net(obs_tensor)[0, action]

        actor_loss = -(q_value * log_prob)
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新折扣因子
        self.discount *= self.gamma if not terminated else 1.0

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

actor_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.001}
critic_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.0002}
agent = QActorCriticAgent(env, actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs)

# 训练
episodes = 150
episode_rewards = []
chart = Chart()
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True, render=True)
    episode_rewards.append(episode_reward)
    print(f"Episode {episode} reward: {episode_reward}")
    chart.plot(episode_rewards)

# 测试
episode_rewards = [play_sarsa(env, agent, render=True) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))