#%%
import numpy as np ,scipy, random
import pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ranksums
from placecell import calculate_global_remapping

# Load place cell activity data
# The data should be a DataFrame with columns representing different neurons
# and rows representing firing rates at different locations for different conditions
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
#%%
results = []

for dd in range(5):
    animal = conddf.animals.values[dd]    
    day = conddf.days.values[dd]    

    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
        'tuning_curves_late_trials', 'coms_early_trials'])
    coms = fall['coms'][0]
    coms_early = fall['coms_early_trials'][0]
    tcs_early = fall['tuning_curves_early_trials'][0]
    tcs_late = fall['tuning_curves_late_trials'][0]
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[dd]    
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    eps = np.append(eps, len(changeRewLoc))  
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)      
        if len(eps)<4: eptest = 2 # if no 3 epochs
    comp = [eptest-2,eptest-1] # eps to compare    

    tc1 = tcs_late[comp[0]]
    tc2 = tcs_late[comp[1]]

    # Calculate the correlation between place fields for the two reward conditions
    data_reward1, data_reward2 = tc1, tc2
    # global remapping calculation
    P, H, real_distribution, shuffled_distribution, p_values, global_remapping = calculate_global_remapping(data_reward1, data_reward2)
    print(f"Rank-sum test p-value: {P}")
    print(f"Percentage of neurons showing global remapping: {np.sum(global_remapping) / len(global_remapping) * 100:.2f}%")
    # Save results
    result_df = pd.DataFrame({
        'Neuron': range(len(real_distribution)),
        'Real_Cosine_Similarity': real_distribution,
        'P_Value': p_values,
        'Global_Remapping': global_remapping
    })

    results.append(result_df)
