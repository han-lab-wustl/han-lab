#%% rnn to predict position
import torch
import torch.nn as nn
import numpy as np, pickle, pandas as pd, scipy, random
import matplotlib.pyplot as plt

track_length = 270
with open("Z:\dcts_com_opto.p", "rb") as fp: #unpickle
        dcts = pickle.load(fp)
dd=4
conddf = pd.read_csv(r"Z:\conddf_neural.csv", index_col=None)

day=conddf.days.values[dd]
animal = conddf.animals.values[dd]
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
fall = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'forwardvel', 'ybinned', 'iscell',
                            'bordercells', 'changeRewLoc'])
inactive = dcts[dd]['inactive']
eptest = conddf.optoep.values[dd]
if conddf.optoep.values[dd]<2: eptest = random.randint(2,3)   
changeRewLoc = np.hstack(fall['changeRewLoc']) 
eps = np.where(changeRewLoc>0)[0]
rewlocs = changeRewLoc[eps]*1.5
eps = np.append(eps, len(changeRewLoc))    
if len(eps)<4: eptest = 2 # if no 3 epochs
comp = [eptest-2,eptest-1] # eps to compare    

# filter iscell
Fc3 = fall['Fc3'][:,(fall['iscell'][:,0].astype(bool)) & (~fall['bordercells'][0].astype(bool))]
neural_activity = Fc3[:, inactive]  
position = fall['ybinned'][0]*1.5
# neural_activity = neural_activity[position>3,:]
# position = position[position>3] # remove dark time
grab = 100
frames = [random.sample(range(eps[xx],eps[xx+1]),grab) for xx in range(len(eps)-1)]
neural_activity_all = np.array([neural_activity[f] for f in frames])
position = fall['ybinned'][0]*1.5
position_chunk = [position[f] for f in frames]
position_chunk = np.array(position_chunk)
# Convert data to PyTorch tensors
neural_activity = torch.from_numpy(neural_activity_all[comp[0]]).float().unsqueeze(1)  # Add a dummy batch dimension
position = torch.from_numpy(position_chunk[comp[0]]).float()

# Define the RNN model
class PositionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(PositionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = len(inactive)  # Number of neurons
hidden_size = 64  # Size of the hidden state
output_size = 1   # Number of output dimensions (1D position)
num_layers = 1    # Number of RNN layers

# Initialize the model
model = PositionDecoder(input_size, hidden_size, 
                    output_size, num_layers)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(neural_activity)
    loss = criterion(outputs, position)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# opto ep compare
neural_activity = torch.from_numpy(neural_activity_all[comp[1]]).float().unsqueeze(1)  # Add a dummy batch dimension
position = torch.from_numpy(position_chunk[comp[1]]).float()
# Make predictions
with torch.no_grad():
    predicted_position = model(neural_activity)
    print('Predicted position:', predicted_position)
    print('Actual position: ', position)

fig,ax = plt.subplots()
ax.plot(position, color='k')
ax.plot(predicted_position, color='orangered')