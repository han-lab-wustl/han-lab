import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
import scipy, pandas as pd, random
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128, output_dim = 1, 
        num_layers = 2):
        super(LSTMModel, self).__init__()
        # Initialize the LSTM, Hidden Layer, and Output Layer
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                dropout = 0.0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        
        return out

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

'''
Training Data is a N*T*n tensor, N is the number of samples, T is the interval,
n is number of neurons.
Training Data is a N*1*1 tensor, N is the number of samples, 1*1 represents the
output dimension, which is the position of last time point.
Testing Data is a N'*T*n tensor.
Testing label is a N*1*1 tensor.
E.g.,
TrainingData = create_subsequences(np.transpose(X, TimeInterval))
TrainingLabel = Y[TimeInterval-1:].reshape(-1,1)
X is dFF, Y is corresponding position.
'''
# import raw data
with open(r"\\storage1.ris.wustl.edu\ebhan\Active\dzahra\analysis\decoding_lstm\dcts_com_opto.p", "rb") as fp: #unpickle
        dcts = pickle.load(fp)
<<<<<<< HEAD
dd=5
conddf = pd.read_csv(r"\\storage1.ris.wustl.edu\ebhan\Active\dzahra\analysis\decoding_lstm\conddf_neural.csv", index_col=None)
=======
dd=0
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural.csv", index_col=None)
>>>>>>> af8cebabd6db6fa3b783487cf35f3eb52b2e2b2b
day=conddf.days.values[dd]
animal = conddf.animals.values[dd]
params_pth = rf"\\storage1.ris.wustl.edu\ebhan\Active\dzahra\analysis\decoding_lstm\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
fall = scipy.io.loadmat(params_pth, variable_names=['dFF', 'forwardvel', 'ybinned', 'iscell',
                            'trialnum', 'bordercells', 'changeRewLoc'])
inactive = dcts[dd]['inactive']
changeRewLoc = np.hstack(fall['changeRewLoc']) 

eptest = conddf.optoep.values[dd]
eps = np.where(changeRewLoc>0)[0]
rewlocs = changeRewLoc[eps]*1.5
<<<<<<< HEAD
eps = np.append(eps, len(changeRewLoc))    
# if len(eps)<4: eptest = 2 # if no 3 epochs
=======
eps = np.append(eps, len(changeRewLoc)) 
if conddf.optoep.values[dd]<2: 
    eptest = random.randint(2,3)   
    if len(eps)<4: eptest = 2 # if no 3 epochs
trialnum = np.hstack(fall['trialnum'])
>>>>>>> af8cebabd6db6fa3b783487cf35f3eb52b2e2b2b
comp = [eptest-2,eptest-1] # eps to compare    
other_ep = [xx for xx in range(len(eps)-1) if xx not in comp]
# filter iscell
dff = fall['dFF'][:,(fall['iscell'][:,0].astype(bool)) & (~fall['bordercells'][0].astype(bool))]
<<<<<<< HEAD
# remove cells with nan?
dff[:,np.where(sum(np.isnan(dff))>0)] = 0
=======
# remove nans
dff[:, sum(np.isnan(dff))>0] = 0
>>>>>>> af8cebabd6db6fa3b783487cf35f3eb52b2e2b2b
dff_per_ep = [dff[eps[xx]:eps[xx+1]] for xx in range(len(eps)-1)]
trialnum_per_ep = [trialnum[eps[xx]:eps[xx+1]] for xx in range(len(eps)-1)]
# get a subset of trials
# normalize from 0 to 1
dff_per_ep_trials = [dff_per_ep[ii][(trialnum_per_ep[ii]>2) & (trialnum_per_ep[ii]!=14) & (trialnum_per_ep[ii]!=13) & (trialnum_per_ep[ii]!=12)] for ii in range(len(eps)-1)]
dff_per_ep_trials_test = [dff_per_ep[ii][(trialnum_per_ep[ii]==14) | (trialnum_per_ep[ii]==13) | (trialnum_per_ep[ii]==12)] for ii in range(len(eps)-1)]
dff_per_ep_trials_norm = [np.array([(xx-np.nanmin(xx))/(np.nanmax(xx)-np.nanmin(xx)) for xx in yy.T]).T for yy in dff_per_ep_trials]
dff_per_ep_trials_test_norm = [np.array([(xx-np.nanmin(xx))/(np.nanmax(xx)-np.nanmin(xx)) for xx in yy.T]).T for yy in dff_per_ep_trials_test]

# normalize by training data
# dff_per_ep_trials_test_norm = [(xx-np.nanmin(dff_per_ep_trials[ii]))/(np.nanmax(dff_per_ep_trials[ii])-np.nanmin(dff_per_ep_trials[ii])) for ii,xx in enumerate(dff_per_ep_trials_test)]
position = fall['ybinned'][0]*1.5
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(position.reshape(-1,1))
# normalize the dataset and print
position = np.squeeze(scaler.transform(position.reshape(-1,1)))

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
TrainingLabel = train_pos[TimeInterval-1:].reshape(-1,1)
test_pos = position_per_ep_trials_test[comp[0]]
TestLabel = test_pos[TimeInterval-1:].reshape(-1,1)

TrainingData = create_subsequences(train,TimeInterval)
batch_size = 64
input_size = TrainingData.shape[-1] # number of cells
output_size = 1

Train_dataset = CreateTimeSeriesData(TrainingData, TrainingLabel)
Train_loader = DataLoader(dataset=Train_dataset, batch_size=batch_size, 
            shuffle=True, drop_last = True)
# Validation_loader = DataLoader(dataset=Train_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
# when predicting, set shuffle = False

TestData = create_subsequences(test,TimeInterval)
Test_dataset = CreateTimeSeriesData(TestData, TestLabel)
Test_loader = DataLoader(dataset=Test_dataset, batch_size=batch_size,
            shuffle=False, drop_last = True)

model = LSTMModel(input_size, output_dim = output_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
criterion = nn.MSELoss()  # For regression tasks
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay = 1e-9)

# Example training loop
l = []
val_l = []
num_epochs = 1000
for epoch in range(num_epochs):
    train_loss = 0.0
    for inputs, targets in Train_loader:
        # Forward pass
        inputs, targets = inputs.to(device).float(), targets.to(device).float()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    l.append(train_loss/len(Train_loader))
    if epoch % 20 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss/len(Train_loader)))
        val_loss = 0.0
        for inputs, targets in Test_loader:
            # Forward pass
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_l.append(val_loss/len(Test_loader))
        print('Validation Loss: {:.4f}'.format(val_loss/len(Test_loader)))

# save
savepth = r'Z:\\model_dd0_conddct.pt'
torch.save(model.state_dict(), savepth)

# use model to predict position
dff_test = dff_per_ep_trials_test_norm[comp[0]]
# TODO: use a couple trials from same ep as testing
# use different epochs for further testing
pos = position_per_ep_trials_test[comp[0]]
pos = pos[TimeInterval-1:].reshape(-1,1)

TimeInterval = 10 # frames
TestData = create_subsequences(dff_test,TimeInterval)
batch_size = 128
input_size = TestData.shape[-1] # number of cells
output_size = 1

Test_dataset = CreateTimeSeriesData(TestData, pos)
Test_loader = DataLoader(dataset=Test_dataset, batch_size=batch_size, 
                        shuffle=False, drop_last = True)
predict = []
for inputs, targets in Test_loader:
    # Forward pass
    inputs, targets = inputs.to(device).float(), targets.to(device).float()
    predict.append(model(inputs))
    loss = criterion(outputs, targets)
    val_loss += loss.item()
    val_l.append(val_loss/len(Test_loader))
print('Validation Loss: {:.4f}'.format(val_loss/len(Test_loader)))