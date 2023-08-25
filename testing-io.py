from scipy.io import loadmat
import numpy as np 

filename = "christos_ODR.mat"
def get_dims(arr):
    if not isinstance(arr, list):
        return []
    return [len(arr)] + get_dims(arr[0])
data = loadmat(filename)
print(data.keys()) 

spiketimes = data["spiketimes"]
spiketimes = spiketimes.reshape(-1)

print(type(spiketimes[0][0]))  # This will print the type of the first element.
print(spiketimes[0][0])       # This will print the actual first element.


print(spiketimes.shape)
print(spiketimes[0].shape)
# If it's a numpy array inside:
print(spiketimes[0][0].shape)

# If it's a list inside:
print(len(spiketimes[0][0]))
