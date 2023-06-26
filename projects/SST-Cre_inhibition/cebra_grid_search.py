# cebra grid search

import numpy as np, os
import cebra
from cebra import CEBRA

mat = [r'Y:\sstcre_imaging\e200\78\230531_ZD_000_000\suite2p\plane0\Fall.mat',
r'Y:\sstcre_imaging\e200\77\230529_ZD_000_000\suite2p\plane0\Fall.mat',
r'Y:\sstcre_imaging\e200\76\230526_ZD_000_000\suite2p\plane0\Fall.mat']
dst = r'Y:\sstcre_analysis\hrz\cebra'

# Load the .npz
dataz = []; posz = []; 
for i in range(3):
    neural_data = cebra.load_data(file=mat[i], 
                key="spks")
    iscell = cebra.load_data(file=mat[i], 
                key="iscell")
    trials = np.squeeze(cebra.load_data(file=mat[i], 
                key="trialnum"))
    rewloc = cebra.load_data(file=mat[i], 
                key="changeRewLoc")
    ybinned = cebra.load_data(file=mat[i], 
                key="ybinned")
    # epoch 1 only
    ep1 = np.where(rewloc>0)[1][1]
    mask = np.arange(0,ep1)
    neural_data = neural_data[iscell[:,0].astype(bool)]    
    pos = np.squeeze(ybinned)[mask][trials[mask]>=3]
    rew = rewloc[rewloc>0][0] # rew location in positiom
    data = neural_data.T[mask][trials[mask]>=3][pos>1]
    pos_ = pos[pos>1]
    dataz.append(data)
    posz.append(pos_)
    
# 1. Define the parameters, either variable or fixed
params_grid = dict(
    output_dimension = [2, 3, 5, 8, 10, 16],
    learning_rate = [0.001],
    time_offsets = 5,
    max_iterations = 100,
    temperature_mode = "auto",
    verbose = False)

# 2. Define the datasets to iterate over
datasets = {"dataset1": dataz[0],                      # time contrastive learning
            "dataset2": (dataz[1], posz[1]), # behavioral contrastive learning
            "dataset3": (dataz[2], posz[2])} # a different set of data

# 3. Create and fit the grid search to your data
grid_search = cebra.grid_search.GridSearch()
grid_search.fit_models(datasets=datasets, params=params_grid, 
                       models_dir=os.path.join(dst, "saved_models"))
# 4. Get the results
df_results = grid_search.get_df_results(models_dir="saved_models")

# 5. Get the best model for a given dataset
best_model, best_model_name = grid_search.get_best_model(dataset_name="dataset2", models_dir="saved_models")
