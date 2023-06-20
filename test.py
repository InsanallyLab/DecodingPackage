
import os

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