import pickle
import matplotlib.pyplot as plt

with open("rewards.pkl", "rb") as f:
    rewards = pickle.load(f)

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards")
plt.show()
