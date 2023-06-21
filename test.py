
import os
import decoding as ilep 
import tqdm 
import multiprocessing as mp

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

REPETITIONS = int(10)  
CATEGORIES = "stimulus"

results_params = []
results_async = []
def log_result(result, final_list=results_async):
    final_list.append(result)

def log_error(x, final_list=results_async):
    final_list.append({})


pool = mp.Pool(mp.cpu_count())

####################################################################################################################################
n_reps = range(REPETITIONS)

EnumSession = []
EnumClust = []
sessions = os.listdir(CACHE_DIR)
for session in sessions:
    sessionfile = ilep.loadSessionCached(CACHE_DIR,session)

    if sessionfile.meta.task == 'passive no behavior':
        continue
    if sessionfile.meta.task in ['tuning nonreversal','tuning switch','tuning reversal']:
        continue
        
    for clust in sessionfile.clusters.good:
        EnumSession.append(session)
        EnumClust.append(clust)

trialsPerDayLoaded = 'NO_TRIM'

####################################################################################################################################

progress_bar = tqdm(zip(EnumSession,EnumClust), desc=f"Calculating {CATEGORIES} decoding")
for session,clust in progress_bar:
    results = {}
    results['n_rep'] = REPETITIONS
    results['categories'] = CATEGORIES
    results['session'] = session
    results['clust'] = clust
    try:
        progress_bar.write(f"{session} cluster {clust} is present.")
        
        #Need to create interval
        trainInterval = ilep.TrialInterval(-0.2*30000,0,False,True)
        testInterval = ilep.TrialInterval(0,0,False,True)
        temp = pool.apply_async(ilep.calculateDecodingForSingleNeuron,(session,clust,trialsPerDayLoaded,CACHE_DIR,OUTPUT_DIR,trainInterval,testInterval,REPETITIONS,CATEGORIES))
        progress_bar.write(f"{session} cluster {clust}: {temp}")
        results_async.append(temp)
        results_params.append(results)

    except Exception as e:
        progress_bar.write(f"{session},{clust}: {e}")
        continue
    
# Closing the worker pool
    pool.close()
    pool.join()

    results_2 = []
    for r in results_async:
        try:
            results_2.append(r.get())
        except Exception as e:
            results_2.append({})
            print(f"problem with async append: {e}")
    
    progress_bar.write(f"results_2: {results_2}")
    progress_bar.write(f"results_async: {results_async}")

    # Combining results
    total_results = [{**d1, **d2} for d1, d2 in zip(results_params, results_2)]

    # Saving as CSV
    results_df = pd.DataFrame(total_results)

    variables=['session','clust']
    results_df = results_df.sort_values(by=variables).reset_index(drop=True)
    results_df.to_csv(os.path.join(pwd, "choicedecoding_opsin_on.csv"))