import os
import decoding as ilep 
from tqdm import tqdm 
import multiprocessing as mp
import time 

import pandas as pd 

# Extract current working directory (pwd)
pwd = os.getcwd()

# Construct CACHE_DIR and OUTPUT_DIR paths
CACHE_DIR = os.path.join(pwd, 'cache')
OUTPUT_DIR = os.path.join(pwd, 'output')

# Create cache and output directories if they don't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

REPETITIONS = 10 
CATEGORIES = "stimulus"

file1 = "TH_234_1_passive_AC.pickle" 

file2 = "BS_85_13_AC.pickle" 

trialsPerDayLoaded = 'NO_TRIM' 
session_file = ilep.loadSessionCached(pwd, file2)

trainInterval = ilep.TrialInterval(-0.2*300000,0,False,True)
testInterval = ilep.TrialInterval(0,0,False,True)

cluster_list = session_file.clusters.good 


cluster1 = cluster_list[0] 


print("The # of clusters is", len(cluster_list))
start_time = time.time()  # Get the current time in seconds


res = ilep.calculateDecodingForSingleNeuron(file2, cluster1,trialsPerDayLoaded,CACHE_DIR,OUTPUT_DIR,trainInterval,testInterval,REPETITIONS,CATEGORIES)
print("The results are: ", res)

print(len(res))
end_time = time.time()  # Get the current time again
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")