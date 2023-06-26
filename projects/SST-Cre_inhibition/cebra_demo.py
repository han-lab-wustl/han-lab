# cebra demo
# zahra

# Create a .npz file
import numpy as np, os

dst = r'Y:\sstcre_analysis\hrz\cebra'
X = np.random.normal(0,1,(100,3))
X_new = np.random.normal(0,1,(100,4))
np.savez(os.path.join(dst,"neural_data"), 
         neural = X, new_neural = X_new)

# Create a .h5 file, containing a pd.DataFrame
import pandas as pd

X_continuous = np.random.normal(0,1,(100,3))
X_discrete = np.random.randint(0,10,(100, ))
df = pd.DataFrame(np.array(X_continuous), columns=["continuous1", "continuous2", "continuous3"])
df["discrete"] = X_discrete
df.to_hdf(os.path.join(dst, "auxiliary_behavior_data.h5"),
        key="auxiliary_variables")

import cebra
from cebra import CEBRA

# Load the .npz
neural_data = cebra.load_data(file=os.path.join(dst,"neural_data.npz"), 
            key="neural")

# ... and similarly load the .h5 file, providing the columns to keep
continuous_label = cebra.load_data(file=os.path.join(dst, "auxiliary_behavior_data.h5"), 
    key="auxiliary_variables", columns=["continuous1", "continuous2", "continuous3"])
discrete_label = cebra.load_data(file=os.path.join(dst, "auxiliary_behavior_data.h5"), 
    key="auxiliary_variables", columns=["discrete"]).flatten()

print(cebra.models.get_options('offset*', limit = 4))

cebra_model = CEBRA(
    model_architecture = "offset10-model",
    batch_size = 1024,
    temperature_mode="auto",
    learning_rate = 0.001,
    max_iterations = 10,
    time_offsets = 10,
    output_dimension = 8,
    device = "cuda_if_available",
    verbose = False
)

print(cebra_model)

timesteps = 5000
neurons = 50
out_dim = 8

neural_data = np.random.normal(0,1,(timesteps, neurons))
continuous_label = np.random.normal(0,1,(timesteps, 3))
discrete_label = np.random.randint(0,10,(timesteps,))

single_cebra_model = cebra.CEBRA(batch_size=512,
                                 output_dimension=out_dim,
                                 max_iterations=10000,
                                 max_adapt_iterations=10)
single_cebra_model.fit(neural_data)
embedding = single_cebra_model.transform(neural_data)
assert(embedding.shape == (timesteps, out_dim))
cebra.plot_embedding(embedding)
cebra.plot_loss(single_cebra_model)
