"""zahra
oct 2024
model to predict neural activity
"""
#%%
import torch, scipy, matplotlib.pyplot as plt, sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')
from rewardcell import get_radian_position

# Load your data (as an example; adjust file paths as needed)
# behavior_data = pd.read_csv('behavior_data.csv')  # Assume columns: ['speed', 'position', 'reward']
# neural_data = pd.read_csv('neural_data.csv')      # Assume columns: ['neuron1', 'neuron2', ...]
animal='e201';day=54;pln=0
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
print(params_pth)
fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 
    'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
    'stat', 'licks', 'timedFF'])
VR = fall['VR'][0][0][()]
scalingf = VR['scalingFACTOR'][0][0]
try:
    rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
except:
    rewsize = 10
ybinned = fall['ybinned'][0]/scalingf
track_length=180/scalingf    
forwardvel = fall['forwardvel'][0]    
changeRewLoc = np.hstack(fall['changeRewLoc'])
trialnum=fall['trialnum'][0]
rewards = fall['rewards'][0]
# set vars
eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
eps = np.append(eps, len(changeRewLoc))

fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
#%%
neural_data = Fc3#dFF[:, skew>2] # time x cells
# smooth
ldf = pd.DataFrame({'lick': fall['licks'][0]})
lick = np.hstack(ldf.rolling(100).mean().fillna(0).values) # 3s 
vdf = pd.DataFrame({'velocity': forwardvel})
velocity = np.hstack(vdf.rolling(20).mean().fillna(0).values) # 3s 
rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
acc = np.append(np.diff(forwardvel)/np.diff(fall['timedFF'][0]),0)
accdf = pd.DataFrame({'acc': acc})
acc = np.hstack(accdf.rolling(50).mean().fillna(0).values)

behavior_data = np.array([ybinned, rad, velocity, acc, lick]).T
# Standardize/normalize the data
behavior_scaler = MinMaxScaler()
neural_scaler = MinMaxScaler()

behavior_data_scaled = behavior_scaler.fit_transform(behavior_data)
neural_data_scaled = neural_scaler.fit_transform(neural_data)
#%%

import statsmodels.api as sm

# split by ep
# half epoch
frame_split=3000
train = np.concatenate([np.arange(eps[xx],np.ceil(eps[xx+1]-frame_split)) for xx in range(len(eps)-1)]).astype(int)
test = np.concatenate([np.arange(np.ceil(eps[xx+1]-frame_split),eps[xx+1]) for xx in range(len(eps)-1)]).astype(int)
# Train-test split

def create_subsequences(time_series, subsequence_length=20):
    num_subsequences = len(time_series) - subsequence_length + 1
    subsequences = [time_series[i:i+subsequence_length] for i in range(num_subsequences)]
    return np.array(subsequences)

seq_length = 20
X = create_subsequences(neural_data_scaled, seq_length)
y = create_subsequences(behavior_data_scaled, seq_length)[:, -1, :]

# Train-test split
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]


# Convert to PyTorch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset objects
train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)

batch_size=256
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#%%
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=5, num_layers=2, batch_first=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state and cell state with correct dimensions
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Only use the output of the last time step
        return out

input_dim = neural_data.shape[1]
output_dim = behavior_data.shape[1]
model = LSTMModel(input_dim=input_dim, output_dim=output_dim).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
        lr=1e-5, weight_decay = 1e-9)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluation loop
model.eval()
total_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        total_loss += loss.item()

print(f'Test Loss: {total_loss/len(test_loader)}')

#%%
# After the training loop, make sure the model is in evaluation mode
model.eval()

# Make predictions on the test set
with torch.no_grad():
    y_pred_test = model(X_test_tensor.cuda()).cpu().detach().numpy()
    y_test_actual = y_test_tensor.cpu().numpy()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize dictionaries to hold metrics for each behavioral variable
rmse = {}
mae = {}
r_squared = {}

# List of behavioral variable names
behavioral_vars = ['ybinned', 'rad', 'velocity', 'acc', 'lick']

# Compute metrics for each behavioral variable
for i, var in enumerate(behavioral_vars):
    rmse[var] = np.sqrt(mean_squared_error(y_test_actual[:, i], y_pred_test[:, i]))
    mae[var] = mean_absolute_error(y_test_actual[:, i], y_pred_test[:, i])
    r_squared[var] = r2_score(y_test_actual[:, i], y_pred_test[:, i])

# Print the metrics
print("Root Mean Squared Error (RMSE):")
print(rmse)
print("\nMean Absolute Error (MAE):")
print(mae)
print("\nR-squared (R2):")
print(r_squared)

#%%
import matplotlib.pyplot as plt

# Plot actual vs. predicted values for each behavioral variable
for i, var in enumerate(behavioral_vars):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual[:, i], label='Actual')
    plt.plot(y_pred_test[:, i], label='Predicted', alpha=0.7)
    plt.title(f'Actual vs Predicted {var}')
    plt.xlabel('Sample Index')
    plt.ylabel(var)
    plt.legend()
    plt.show()
    
#%%import torch
import numpy as np
from captum.attr import FeatureAblation

# Define a simplified predict_lick function to extract the specific output (lick) we're interested in
def predict_lick(input):
    model.eval()
    with torch.no_grad():
        output = model(input)
        return output[:,4]

# Initialize Feature Ablation
fa = FeatureAblation(predict_lick)

# Evaluate on a subset of the test set for faster computation
subset_X_test = X_test_tensor[:10000]  # Taking a subset of the first 100 samples

# Compute attributions using Feature Ablation
attributions = fa.attribute(subset_X_test.cuda(), feature_mask=None)
attributions = attributions.cpu().detach().numpy()

# Summarize by averaging over sequences and samples
average_lick_attributions = np.mean(np.abs(attributions), axis=(0, 1))

# Print the average attributions for each neuron
for i, attribution in enumerate(average_lick_attributions):
    print(f"Neuron {i+1}: {attribution:.4f}")
#%%
thres=0.0002
average_lick_attributions_mask = average_lick_attributions[average_lick_attributions>thres]
n = np.where(average_lick_attributions>thres)[0]
# Plot average attributions
plt.figure(figsize=(10, 5))
plt.barh([f'Neuron {i+1}' for i in n], 
        average_lick_attributions_mask)
plt.xlabel('Average Feature Ablation Importance')
plt.title('Neuron Contribution to Predicting Lick Behavioral Variable')
plt.show()

# %%

#%%
# test neuron contribution
rng = np.arange(0,20000)
for nr in n:
    plt.figure()
    plt.plot(neural_data[rng,nr])
    plt.plot(lick[rng])