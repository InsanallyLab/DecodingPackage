import util 
from types import SimpleNamespace 
import numpy as np 
from KDEpy import FFTKDE 
from analysis import *
from scipy.interpolate import interp1d 


################################### Training ##################################

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

def synthetic_spiketrain(model,trial_length=2500):#only model.all is required to be filled out. Units in ms
    LogISIs = []
    ctime = 0
    while True:
        ISI = model.all.Inv_Likelihood(np.random.rand())
        ctime += 10**ISI
        if ctime <= trial_length:
            LogISIs.append(ISI)
        else:
            break
    return np.array(LogISIs)

def cachedpredictTrial(sessionfile,clust,model,trialISIs,conditions = ['target_tone','nontarget_tone'],synthetic = False):
    if synthetic:
        LogISIs = synthetic_spiketrain(model)
    else:
        LogISIs = trialISIs
    
    #Set up probabilities with initial values equal to priors
    probabilities = dict()
    for cond in conditions:
        probabilities[cond] = SimpleNamespace()
        probabilities[cond].prob = np.full(len(LogISIs)+1,np.nan)
        probabilities[cond].prob[0] =  np.log10(model.conds[cond].Prior_0)
    
    #Calculate change in W. LLR after each LogISI
    for cond in conditions:
        probabilities[cond].prob = np.cumsum(np.log10(np.concatenate((   [model.conds[cond].Prior_0] , model.conds[cond].Likelihood(LogISIs)   ))))

    ##Calculate change in W. LLR after each LogISI
    #for idx,logISI in enumerate(LogISIs):
    #    for cond in conditions:
    #        probabilities[cond].prob[idx+1] = probabilities[cond].prob[idx] + np.log10(model[cond].Likelihood.evaluate(logISI))

    #Exponentiate back from log so that normalization can take place
    for cond in conditions:
        probabilities[cond].prob = np.power(10,probabilities[cond].prob)
    
    #Calculate total probability sum to normalize against
    sum_of_probs = np.zeros(len(LogISIs)+1)  
    for cond in conditions:
        sum_of_probs += probabilities[cond].prob

    #Normalize all probabilities
    for cond in conditions:
        probabilities[cond].prob /= sum_of_probs


    #No ISIs in trial. guess instead.
    if len(LogISIs) < 1:
        keys = [cond for cond in probabilities]
        probs = [model.conds[cond].Prior_empty for cond in probabilities]
        maxidx = np.concatenate(np.argwhere(np.equal(probs,np.max(probs))))
        if len(maxidx) > 1:
            maxidx = np.random.permutation(maxidx)
        maxidx = maxidx[0]
        maxCond = keys[maxidx]
        #return None,None,None,None
        return maxCond,probs[maxidx],probabilities,True
    #ISIs were present in the trial. We can take out a proper choice by the algorithm.
    else:
        keys = [cond for cond in probabilities]
        probs = [probabilities[cond].prob for cond in probabilities]
        probs = [p[len(p)-1] for p in probs]
        maxidx = np.argmax(probs)
        maxCond = keys[maxidx]
        return maxCond,probs[maxidx], probabilities,False
        #return maxCond,probabilities[maxCond].prob[len(times)-1], probabilities,times

def cachedtrainDecodingAlgorithm(loader,clust,trialsPerDayLoaded,bw,cachedLogISIs,Train_X,condition_names,synthetic = False):
    model = SimpleNamespace()
    model.conds = dict()
    model.all = SimpleNamespace()
    for cond in condition_names:
        model.conds[cond] = SimpleNamespace()
    
    #Determine trials to use for each condition. Uses the conditions structures from
    #ilep to increase modularity and to ensure that all code is always using the same
    #conditions
    decoding_conditions = splitByConditions(loader,clust,trialsPerDayLoaded,Train_X,condition_names)
    
    #Handle all_trials
    #print(f"Starting model training.\nThere are {len(Train_X)} trials")
    LogISIs = cachedLogISIs[Train_X]
    #print(f"Starting model training.\nThere are {len(LogISIs)} trial_ISIs")
    if len(LogISIs)<5:
        #print(f"Skipping fold. Not enough trials for ISI concatenation")
        return None
    LogISIs = np.concatenate(LogISIs)
    #print(f"Starting model training.\nThere are {len(LogISIs)} ISIs")
    #Check for feasability of this model
    if len(LogISIs)<5:
        #print(f"Skipping fold. Not enough ISIs")
        return None

    f,inv_f = LogISIsToLikelihoods(LogISIs,bw)
    model.all.Likelihood = f
    model.all.Inv_Likelihood = inv_f

    #Handle synthetic spiketrain generation
    if synthetic:
        cachedLogISIs = [ [] ]*loader.meta.length_in_trials
        for trial in Train_X:
            cachedLogISIs[trial] = synthetic_spiketrain(model)
        cachedLogISIs = np.array(cachedLogISIs,dtype='object')

    #Handle individual conditions
    LogISIs_per_trial = [len(l) for l in cachedLogISIs[Train_X]]
    total_empty_ISI_trials = np.sum(np.equal(  LogISIs_per_trial,0  ))
    for cond in decoding_conditions:        
        
        LogISIs = cachedLogISIs[decoding_conditions[cond].trials]
        LogISIs = np.concatenate(LogISIs)

        if len(LogISIs)<5:
            print(f"Skipping fold. Not enough ISIs for the {cond} condition")
            return None

        #LogISIs,_ = getLogISIs(sessionfile,clust,decoding_conditions[cond].trials)

        f,inv_f = LogISIsToLikelihoods(LogISIs,bw) #This is a gaussian KDE
        model.conds[cond].Likelihood = f
        model.conds[cond].Inv_Likelihood = inv_f
        model.conds[cond].Prior_0 = 1.0 / len(condition_names)

        #Calculate likelihood for 0-ISI trials
        LogISIs_per_trial = [len(l) for l in cachedLogISIs[decoding_conditions[cond].trials]]
        numerator = np.sum(np.equal(LogISIs_per_trial,0)) + 1
        denominator = total_empty_ISI_trials + len(condition_names)
        model.conds[cond].Prior_empty = numerator / denominator
        
    return model

def cachedcalculateAccuracyOnFold(sessionfile,clust,model,cachedLogISIs,Test_X,weights,conditions=['target','nontarget'],synthetic=False):
    #Note: this call to getAllConditions uses NO_TRIM all the time because it is only used to determine correctness
    all_conditions = getAllConditions(sessionfile,clust,trialsPerDayLoaded='NO_TRIM')
    
    accumulated_correct = 0
    #accumulated_total = 0 # I have changed how weighting works to be weighted by condition prevalence so this is no longer necessary
    num_correct = 0
    num_total = 0
    num_empty = 0
    
    for trial in Test_X:
        cond,_,_,empty_ISIs = cachedpredictTrial(sessionfile,clust,model,cachedLogISIs[trial],conditions=conditions,synthetic=synthetic)
        
        if cond is None:
            continue

        if np.isin(trial,all_conditions[cond].trials):
            accumulated_correct += weights[cond]
            num_correct += 1
        #accumulated_total += prob
        num_total += 1

        if empty_ISIs:
            num_empty += 1
        
    if num_total <= 0:
        return np.nan,np.nan,np.nan
    else:
        #print(f"Successful calculation of accuracy")
        return (num_correct/num_total),(accumulated_correct/num_total),(num_empty/num_total)
