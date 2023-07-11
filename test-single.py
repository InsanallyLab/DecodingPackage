import os
from tqdm import tqdm 
import multiprocessing as mp
import time 
from decoder.main import * 
from decoder.io.load import *
import pandas as pd 

# Extract current working directory (pwd)
pwd = os.getcwd()

OUTPUT_DIR = os.path.join(pwd, 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

REPETITIONS = 10 
CATEGORIES = "stimulus"

#folder_path = "/Users/insanallylab/Desktop/Priyanshi/Analysis_Cache/"

folder_path = "/Users/priyanshigarg/Downloads/Analysis_Cache" 
file1 = "TH_234_1_passive_AC.pickle" 

file2 = "BS_85_13_AC.pickle" 

test_files = [ "BS_56_1_AC.pickle", "BS_56_2_AC.pickle", "LA_204_1_passive_AC.pickle", "DS_16_4_M2.pickle"]

with open(os.path.join(folder_path,test_files[3]), 'rb') as f:
        session = pickle.load(f)

trialsPerDayLoaded = 'NO_TRIM' 

trainInterval = TrialInterval(-0.2*30000,0,False,True)
testInterval = TrialInterval(0,0,False,True)

loader = LoadSession(session)

decoder = Decoder(loader) 

cluster_list = session.clusters.good

cluster1 = cluster_list[0] 

start_time = time.time()  # Get the current time in seconds

res = decoder.calculateDecodingForSingleNeuron(cluster1,trialsPerDayLoaded,OUTPUT_DIR,trainInterval,testInterval,REPETITIONS,CATEGORIES)
end_time = time.time()  # Get the current time again
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")