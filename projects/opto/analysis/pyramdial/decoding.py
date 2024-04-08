import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
import scipy, pandas as pd, random

# Assuming you have the following data:
# X_train: 3D tensor of shape (num_samples, 10, num_features)
# y_train: 1D tensor of shape (num_samples,) containing the y-coordinate of the position

# Define the multi-layer LSTM model
# class YPositionPredictionModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(YPositionPredictionModel, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         _, (h_n, c_n) = self.lstm(x)
#         output = self.fc(h_n[-1])
#         return output.squeeze(-1)
    
class YPositionPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(YPositionPredictionModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        output = self.fc(h_n[-1])
        return output.squeeze(-1)


def create_subsequences(time_series, subsequence_length=20):
    num_subsequences = len(time_series) - subsequence_length + 1
    subsequences = [time_series[i:i+subsequence_length] for i in range(num_subsequences)]
    return np.array(subsequences)

class CreateTimeSeriesData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

# import raw data
with open("Z:\dcts_com_opto.p", "rb") as fp: #unpickle
        dcts = pickle.load(fp)
dd=0
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural.csv", index_col=None)
day=conddf.days.values[dd]
animal = conddf.animals.values[dd]
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
fall = scipy.io.loadmat(params_pth, variable_names=['dFF', 'forwardvel', 'ybinned', 'iscell',
                            'trialnum', 'bordercells', 'changeRewLoc'])
inactive = dcts[dd]['inactive']
changeRewLoc = np.hstack(fall['changeRewLoc']) 
dct = dcts[dd]
eptest = conddf.optoep.values[dd]
eps = np.where(changeRewLoc>0)[0]
rewlocs = changeRewLoc[eps]*1.5
eps = np.append(eps, len(changeRewLoc)) 
if conddf.optoep.values[dd]<2: 
    eptest = random.randint(2,3)   
    if len(eps)<4: eptest = 2 # if no 3 epochs
trialnum = np.hstack(fall['trialnum'])
comp = [eptest-2,eptest-1] # eps to compare    

# filter iscell
dff = fall['dFF'][:,(fall['iscell'][:,0].astype(bool)) & (~fall['bordercells'][0].astype(bool))]
# remove nans
dff[:, sum(np.isnan(dff))>0] = 0
dff_per_ep = [dff[eps[xx]:eps[xx+1]] for xx in range(len(eps)-1)]
trialnum_per_ep = [trialnum[eps[xx]:eps[xx+1]] for xx in range(len(eps)-1)]
# get a subset of trials
# normalize from 0 to 1
dff_per_ep_trials = [dff_per_ep[ii][(trialnum_per_ep[ii]>2) & (trialnum_per_ep[ii]!=14) & (trialnum_per_ep[ii]!=13) & (trialnum_per_ep[ii]!=12)] for ii in range(len(eps)-1)]
dff_per_ep_trials_test = [dff_per_ep[ii][(trialnum_per_ep[ii]==14) | (trialnum_per_ep[ii]==13) | (trialnum_per_ep[ii]==12)] for ii in range(len(eps)-1)]
dff_per_ep_trials_norm = [(xx-np.nanmin(xx))/(np.nanmax(xx)-np.nanmin(xx)) for xx in dff_per_ep_trials]
# normalize by training data
dff_per_ep_trials_test_norm = [(xx-np.nanmin(dff_per_ep_trials[ii]))/(np.nanmax(dff_per_ep_trials[ii])-np.nanmin(dff_per_ep_trials[ii])) for ii,xx in enumerate(dff_per_ep_trials_test)]
position = fall['ybinned'][0]*1.5
position_per_ep = [position[eps[xx]:eps[xx+1]] for xx in range(len(eps)-1)]
# get a subset of trials
position_per_ep_trials = [position_per_ep[ii][(trialnum_per_ep[ii]>2) & (trialnum_per_ep[ii]!=14) & (trialnum_per_ep[ii]!=13) & (trialnum_per_ep[ii]!=12)] for ii in range(len(eps)-1)]
position_per_ep_trials_test = [position_per_ep[ii][(trialnum_per_ep[ii]==14) | (trialnum_per_ep[ii]==13) | (trialnum_per_ep[ii]==12)] for ii in range(len(eps)-1)]

TimeInterval = 10 # frames
train = dff_per_ep_trials[comp[0]]
test = dff_per_ep_trials_test[comp[0]]
# TODO: use a couple trials from same ep as testing
# use different epochs for further testing
train_pos = position_per_ep_trials[comp[0]]
test_pos = position_per_ep_trials_test[comp[0]]
TrainingData = create_subsequences(train,TimeInterval)
TestData = create_subsequences(test,TimeInterval)
TrainingLabel = train_pos[TimeInterval-1:].reshape(-1,1)
TestLabel = test_pos[TimeInterval-1:].reshape(-1,1)

batch_size = 128
input_size = TrainingData.shape[-1] # number of cells
output_size = 1
# Convert data to PyTorch tensors
TrainingData = torch.tensor(TrainingData, dtype=torch.float32)
TrainingLabel = torch.tensor(TrainingLabel, dtype=torch.float32)


# Train_dataset = CreateTimeSeriesData(TrainingData, TrainingLabel)
# when predicting, set shuffle = False
# TestData = create_subsequences(test,TimeInterval)
# Test_dataset = CreateTimeSeriesData(TestData, TestLabel)
    
# Instantiate the model
model = YPositionPredictionModel(input_size=input_size, 
            hidden_size=128, num_layers=1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
num_epochs = 3000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(TrainingData.float())
    loss = criterion(outputs, TrainingLabel)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

test = dff_per_ep_trials_test[comp[1]]
# TODO: use a couple trials from same ep as testing
# use different epochs for further testing
test_pos = position_per_ep_trials_test[comp[1]]
TestData = create_subsequences(test,TimeInterval)
TestLabel = test_pos[TimeInterval-1:].reshape(-1,1)


# Predict the Y position for new neural activity data
X_new = torch.tensor(TestData, dtype=torch.float32) # 10 frames of neural activity
predicted_y_position = model(X_new)
print(predicted_y_position)