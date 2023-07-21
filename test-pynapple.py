import os
import sys 
import pickle
# adding Folder_2 to the system path
sys.path.insert(0, '/Users/priyanshigarg/Desktop/insanallyLabs/DecodingPackage/decoder/io') 

from load import * 
import multiprocessing as mp
import pandas as pd 
import pynapple as nap 

#folder_path = "/Users/insanallylab/Desktop/Priyanshi/Analysis_Cache/"

folder_path = "/Users/priyanshigarg/Downloads/Analysis_Cache" 
file1 = "TH_234_1_passive_AC.pickle" 

file2 = "BS_85_13_AC.pickle" 

test_files = [ "BS_56_1_AC.pickle", "BS_56_2_AC.pickle", "LA_204_1_passive_AC.pickle", "DS_16_4_M2.pickle"] 
path = folder_path + test_files[1]

data_directory = folder_path

# LOADING DATA
data = nap.load_session(data_directory)

print(type(data)) 

print(dir(data))

with open(os.path.join(folder_path,test_files[3]), 'rb') as f:
        session = pickle.load(f)

data = LoadSession(session) 


print(data.spikes)

print(data.trials)