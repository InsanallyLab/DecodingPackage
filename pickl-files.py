
folder_path = "/Users/insanallylab/Desktop/priyanshi/Analysis_Cache/"
#file_name = "TH_234_1_passive_AC.pickle"
file_name  = "LA_204_7_passive_AC.pickle"
import pickle

def explore_namespace(obj, indent=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{indent}{key} ({type(value).__name__}):")
            explore_namespace(value, indent + "  ")
    elif isinstance(obj, list):
        for item in obj:
            print(f"{indent}[{type(item).__name__}]:")
            explore_namespace(item, indent + "  ")
    elif hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            print(f"{indent}{key} ({type(value).__name__}):")
            explore_namespace(value, indent + "  ")

# Load the pickled data
with open(folder_path + file_name, 'rb') as f:
    data = pickle.load(f)

# Explore the pickled namespace
#explore_namespace(data)
explore_namespace(data.trials)

# print(type(data.trials.response[0])) 
# print(type(data.trials.go[0]))
# print(type(data.trials.starts[0]))
# print(type(data.trials.target[0]))
# print(hasattr(data.trials, "laser_stimulation"))

# print(data.spikes)



