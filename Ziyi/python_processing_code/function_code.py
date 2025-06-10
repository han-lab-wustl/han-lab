# putting some functions here
def find_start_points(data):
    # Check if the first element is 1
    starts = [0] if data[0] == 1 else []
    
    # Find the indices where data changes from 0 to 1
    changes = np.where(np.diff(data) == 1)[0]
    
    # Since np.diff reduces the length by 1, add 1 to each index to get the actual start points
    starts.extend(changes + 1)
    
    return starts
