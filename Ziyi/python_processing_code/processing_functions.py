
import numpy as np  # It's good practice to import at the top of the file

def find_start_points(data):
    """
    Find the starting points of 1's in a binary sequence where the previous element is 0.
    
    Parameters:
    - data (np.array): A numpy array of binary values (0s and 1s).
    
    Returns:
    - list: A list of indices indicating where 1's sequences start.
    """
    # Optionally check if input data is a numpy array
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")

    # Initialize the list for start points
    starts = []

    # Check if the first element is 1 and there is no preceding element
    if data[0] == 1:
        starts.append(0)

    # Find the indices where data changes from 0 to 1
    changes = np.where(np.diff(data) == 1)[0]

    # Since np.diff reduces the length by 1, add 1 to each index to get the actual start points
    starts.extend(changes + 1)

    return starts
