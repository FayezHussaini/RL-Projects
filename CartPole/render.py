import gymnasium as gym
import torch
from model import DQN

env = gym.make("CartPole-v1", render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("dqn_cartpole.pth"))
model.eval()

state, _ = env.reset()
done = False

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = torch.argmax(model(state_tensor)).item()

    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
