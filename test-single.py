import os
import decoding as ilep 
from tqdm import tqdm 
import multiprocessing as mp

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

file = "TH_234_1_passive_AC.pickle"
trialsPerDayLoaded = 'NO_TRIM' 
session_file = ilep.loadSessionCached(pwd, file)


trainInterval = ilep.TrialInterval(-0.2*30,0,False,True)

testInterval = ilep.TrialInterval(0,0,False,True)


cluster_list = session_file.clusters.good 

print("This is the list of good clusters: ", cluster_list)


for cluster in cluster_list: 
    print("current cluser: ", cluster)
    res = ilep.calculateDecodingForSingleNeuron(file,cluster,trialsPerDayLoaded,CACHE_DIR,OUTPUT_DIR,trainInterval,testInterval,REPETITIONS,CATEGORIES)
    print(res)