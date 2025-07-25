import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

# -----------------------------
# Example dummy data (replace with real)
# -----------------------------
T = 10000  # number of time points
N = 50     # number of neurons

np.random.seed(42)
neural_activity = np.random.rand(T, N)
position = np.linspace(0, 250, T) + np.random.randn(T) * 5  # add jitter
lick = np.zeros(T)
lick[::500] = 1  # simulate periodic licks

reward_pos = 200
pre_reward_window = 40  # cm before reward

# -----------------------------
# Create label: lick-in-pre-reward-zone
# -----------------------------
in_pre_reward = (position >= reward_pos - pre_reward_window) & (position < reward_pos)
lick_in_pre_reward = (lick == 1) & in_pre_reward

# Shift label earlier to predict the future (e.g., 1s ahead)
lag = 20  # assuming 20 frames = 1 sec
