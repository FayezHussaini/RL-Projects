import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
from model import DQN
import pickle

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

memory = deque(maxlen=10000)

batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 500
rewards_history = []

def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return torch.argmax(policy_net(state)).item()

def train_step():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    current_q = policy_net(states).gather(1, actions)
    next_q = target_net(next_states).max(1)[0].unsqueeze(1)

    target_q = rewards + gamma * next_q * (1 - dones)

    loss = torch.nn.functional.mse_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_step()

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    rewards_history.append(total_reward)

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward}")

torch.save(policy_net.state_dict(), "dqn_cartpole.pth")

with open("rewards.pkl", "wb") as f:
    pickle.dump(rewards_history, f)
