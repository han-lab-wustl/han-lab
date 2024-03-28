import numpy as np, scipy
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
# Assuming your data is in the shape (neurons, time_points)
animal = 'e218'
day = 35
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
fall = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'iscell'])
neural_data = fall['Fc3'][:,fall['iscell'][:,0].astype(bool)]
neural_data = np.nan_to_num(neural_data, nan=0.0) # correct nans
neural_data[np.isinf(neural_data)] = 0.0

# Create an instance of the FastICA algorithm
ica = FastICA(n_components=10, random_state=42)
# Fit the ICA model to the neural data
fit_n = ica.fit(neural_data)  # Transpose the data to have (time_points, neurons) shape
# Get the independent components
independent_components = ica.components_
# Get the mixing matrix
mixing_matrix = ica.mixing_
# Transform the data to the independent component space
neural_data_ica = ica.transform(neural_data).T
# Print the independent components
print("Independent Components:")
print(independent_components)

# Print the mixing matrix
print("\nMixing Matrix:")
print(mixing_matrix)

ica_embedding = ica.fit_transform(neural_data)

# Plot the ICA embedding
plt.figure(figsize=(8, 6))
plt.scatter(ica_embedding[:, 0], ica_embedding[:, 1], s=5)
plt.title('ICA Embedding')
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
plt.show()

