import util 
from types import SimpleNamespace 
import numpy as np 
from KDEpy import FFTKDE 
from scipy.interpolate import interp1d 


################################### Training ##################################

#Trials is passed in 
def splitByConditions(sessionfile,clust,trialsPerDayLoaded,trials,condition_names):
    all_conditions = util.getAllConditions(sessionfile,clust,trialsPerDayLoaded=trialsPerDayLoaded)
    
    decoding_conditions = dict()
    for cond in condition_names:
        decoding_conditions[cond] = SimpleNamespace()
    
    for cond in decoding_conditions:
        condition_trials = trials[np.isin(trials, all_conditions[cond].trials )]
        decoding_conditions[cond].trials = condition_trials
        
    return decoding_conditions
 
def LogISIsToLikelihoods(LogISIs, bw):
    x = np.linspace(-2,6,100)
    y = FFTKDE(bw=bw, kernel='gaussian').fit(LogISIs, weights=None).evaluate(x)
    
    # Use scipy to interpolate and evaluate on arbitrary grid
    f = interp1d(x, y, kind='linear', assume_sorted=True)

    #Also want to generate an inverse CDF for sampling
    #This should go [0,1] -> xrange
    norm_y = np.cumsum(y) / np.sum(y)
    norm_y[0] = 0
    norm_y[len(norm_y)-1] = 1
    inv_f = interp1d(norm_y,x, kind='linear', assume_sorted=True)

    return f,inv_f
