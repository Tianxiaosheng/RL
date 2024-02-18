import numpy as np
np.random.seed(0)
import scipy.optimize
import gym

def optimize_bellman(env, gamma=1.):
    p = np.zeros((env.nS, env.nA, env.nS))
    r = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for prob, next_state, reward, terminated in env.P[state][action]:
                p[state, action, next_state] += prob
                r[state, action] += (reward * prob)
    c = np.ones(env.nS)
    a_ub = gamma * p.reshape(-1, env.nS) - \
            np.repeat(np.eye(env.nS), env.nA, axis=0)
    b_ub = -r.reshape(-1)
    bounds = [(None, None),] * env.nS
    res = scipy.optimize.linprog(c, a_ub, b_ub, bounds=bounds, method='highs')
    v = res.x
    q = r + gamma * np.dot(p, v)
    return v, q


env = gym.make('CliffWalking-v0')
print('观测空间 = {}'.format(env.observation_space))
print('action_space = {}'.format(env.action_space))
print('state size = {}, action size = {}'.format(env.nS, env.nA))
print('map size = {}'.format(env.shape))

optimize_state_values, optimize_action_values = optimize_bellman(env)
print('optimize state value = {}'.format(optimize_state_values))
print('optimize action value = {}'.format(optimize_action_values))

optimal_actions = optimize_action_values.argmax(axis=1)
print("optimize action = {}".format(optimal_actions))