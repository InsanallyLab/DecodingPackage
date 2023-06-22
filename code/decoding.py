
from itertools import product 
import numpy as np
import pickle
import os
from types import SimpleNamespace
from scipy.stats import gaussian_kde, sem, mannwhitneyu
from scipy.interpolate import interp1d
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.neighbors import KernelDensity
from KDEpy import FFTKDE
import util 
from validation import K_fold_strat 
import bandwidth


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
    #minLogISI = 0 #Set minimum to 1ms. This is shorter than the refractory period so we can guarantee we won't get any ISIs less than this
    #maxLogISI = np.max(LogISIs) +1 #Set this to 10x the maximum ISI. This nearly guarantees that there won't be any ISIs outside this range
    
    #KDE = gaussian_kde(LogISIs,bw_method=bw)
    #return KDE

    x = np.linspace(-2,6,100)
    y = FFTKDE(bw=bw, kernel='gaussian').fit(LogISIs, weights=None).evaluate(x)
    
    # Use scipy to interplate and evaluate on arbitrary grid
    f = interp1d(x, y, kind='linear', assume_sorted=True)

    #Also want to generate an inverse CDF for smapling
    #This should go [0,1] -> xrange
    norm_y = np.cumsum(y) / np.sum(y)
    norm_y[0] = 0
    norm_y[len(norm_y)-1] = 1
    inv_f = interp1d(norm_y,x, kind='linear', assume_sorted=True)

    return f,inv_f




def cacheLogISIs(sessionfile,clust,interval):#Include conditions
    ISIs = []
    for trial in range(sessionfile.meta.length_in_trials):
        starttime,endtime = interval._ToTimestamp(sessionfile,trial)
        spiketimes = util.getSpikeTimes(sessionfile,clust=clust,starttime = starttime, endtime = endtime)
        spiketimes = spiketimes * 1000 / sessionfile.meta.fs

        ISIs.append(np.diff(spiketimes))

    LogISIs = np.array([np.log10(tISIs) for tISIs in ISIs],dtype='object')
    return LogISIs

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

def cachedtrainDecodingAlgorithm(sessionfile,clust,trialsPerDayLoaded,bw,cachedLogISIs,Train_X,condition_names,synthetic = False):
    model = SimpleNamespace()
    model.conds = dict()
    model.all = SimpleNamespace()
    for cond in condition_names:
        model.conds[cond] = SimpleNamespace()
    
    #Determine trials to use for each condition. Uses the conditions structures from
    #ilep to increase modularity and to ensure that all code is always using the same
    #conditions
    decoding_conditions = splitByConditions(sessionfile,clust,trialsPerDayLoaded,Train_X,condition_names)
    
    #Handle all_trials
    LogISIs = cachedLogISIs[Train_X]
    LogISIs = np.concatenate(LogISIs)
    f,inv_f = LogISIsToLikelihoods(LogISIs,bw)
    model.all.Likelihood = f
    model.all.Inv_Likelihood = inv_f

    #Handle synthetic spiketrain generation
    if synthetic:
        cachedLogISIs = [ [] ]*sessionfile.meta.length_in_trials
        for trial in Train_X:
            cachedLogISIs[trial] = synthetic_spiketrain(model)
        cachedLogISIs = np.array(cachedLogISIs,dtype='object')

    #Handle individual conditions
    LogISIs_per_trial = [len(l) for l in cachedLogISIs[Train_X]]
    total_empty_ISI_trials = np.sum(np.equal(  LogISIs_per_trial,0  ))
    for cond in decoding_conditions:        
        
        LogISIs = cachedLogISIs[decoding_conditions[cond].trials]
        LogISIs = np.concatenate(LogISIs)

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
    
    #Calculate change in Weighted Log Likelihood ratio after each LogISI
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

def cachedcalculateAccuracyOnFold(sessionfile,clust,model,cachedLogISIs,Test_X,weights,conditions=['target','nontarget'],synthetic=False):
    #Note: this call to getAllConditions uses NO_TRIM all the time because it is only used to determine correctness
    all_conditions = util.getAllConditions(sessionfile,clust,trialsPerDayLoaded='NO_TRIM')
    
    accumulated_correct = 0
    #accumulated_total = 0 # I have changed how weighting works to be weighted by condition prevalence so this is no longer necessary
    num_correct = 0
    num_total = 0
    num_empty = 0
    
    for trial in Test_X:
        cond,prob,_,empty_ISIs = cachedpredictTrial(sessionfile,clust,model,cachedLogISIs[trial],conditions=conditions,synthetic=synthetic)
        
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
        return (num_correct/num_total),(accumulated_correct/num_total),(num_empty/num_total)

def calculateDecodingForSingleNeuron(session,clust,trialsPerDayLoaded,cache_directory,output_directory,trainInterval,testInterval,reps = 1,categories='stimulus'): 

    sessionfile = util.loadSessionCached(cache_directory,session)
    filename = util.generateDateString(sessionfile) + ' cluster ' + str(clust) + ' decoding cached result.pickle'
    filename = os.path.join(output_directory,filename)
    
    try:    
        #a,astd,asem,wa,wastd,wasem,ca,castd,casem,pval = ilep.cachedCalculateClusterAccuracy(sessionfile,clust,reps=reps,categories='stimulus')
        trainInterval._CalculateAvgLickDelay(sessionfile)
        testInterval._CalculateAvgLickDelay(sessionfile)
        res = cachedCalculateClusterAccuracy(sessionfile,clust,trialsPerDayLoaded,trainInterval,testInterval,reps=reps,categories=categories)
    except Exception as e:
        res = generateNullResults()
        print(f"session {session} cluster {clust} has thrown error {e}")
        raise e
    try:
        with open(filename, 'wb') as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Problem saving {f} to {filename}. Error: {e}")
            
    print(f"finished with {util.generateDateString(sessionfile)} cluster {clust}")
    return res

def calculate_weights(sessionfile,clust,trimmed_trials_active,categories,trialsPerDayLoaded=None):
    num_total_trials = len(trimmed_trials_active)
    all_conditions = util.getAllConditions(sessionfile,clust,trialsPerDayLoaded=trialsPerDayLoaded)

    weights = dict()
    for cat in categories:
        weights[cat] = len(all_conditions[cat].trials)/num_total_trials
        weights[cat] = (1 / weights[cat]) / len(categories)
    return weights

def cachedCalculateClusterAccuracy(sessionfile,clust,trialsPerDayLoaded,trainInterval,testInterval,reps = 1,categories='stimulus',bw=None,K_fold_num=10):
    if bw is None:
        #bwInterval = TrialInterval(0,2.5*sessionfile.meta.fs,False,False)
        print("entering grid search")
        best_bw = bandwidth.sklearn_grid_search_bw(sessionfile,clust,trialsPerDayLoaded,trainInterval)
        print("exited grid search")
    else:
        best_bw = bw
        
    accuracy_per_fold = []
    waccuracy_per_fold = []
    accuracy_std_per_fold = []
    waccuracy_std_per_fold = []
    accuracy_sem_per_fold = []
    waccuracy_sem_per_fold = []
    
    control_accuracy_per_fold = []
    control_accuracy_std_per_fold = []
    control_accuracy_sem_per_fold = []
    
    synthetic_accuracy_per_fold = []
    synthetic_accuracy_std_per_fold = []
    synthetic_accuracy_sem_per_fold = []

    fraction_empty = []

    if categories == 'stimulus':
        categories = ['target','nontarget']
    elif categories == 'response':
        categories = ['go','nogo']

    elif categories == 'stimulus_off':
        categories = ['laser_off_target','laser_off_nontarget']
    elif categories == 'stimulus_on':
        categories = ['laser_on_target','laser_on_nontarget']
    elif categories == 'response_off':
        categories = ['laser_off_go','laser_off_nogo']
    elif categories == 'response_on':
        categories = ['laser_on_go','laser_on_nogo']
    
    cachedTrainLogISIs = cacheLogISIs(sessionfile,clust,trainInterval)
    cachedTestLogISIs = cacheLogISIs(sessionfile,clust,testInterval)

    model = None
    model_c = None #Condition permutation
    model_s = None #Synthetic spiketrains

    #Get active trimmed trials
    trimmed_trials_active = np.array(sessionfile.trim[clust].trimmed_trials)
    if trialsPerDayLoaded != 'NO_TRIM':
        active_trials = trialsPerDayLoaded[sessionfile.meta.animal][sessionfile.meta.day_of_training]
        trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)]

    #Remove all trials that do not belong to one of the conditions in question
    included_in_conditions_mask = []
    all_conditions = util.getAllConditions(sessionfile,clust,trialsPerDayLoaded=trialsPerDayLoaded)
    for category in categories:
        included_in_conditions_mask = np.concatenate((included_in_conditions_mask,all_conditions[category].trials))
    trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,included_in_conditions_mask)]

    weights = calculate_weights(sessionfile,clust,trimmed_trials_active,categories,trialsPerDayLoaded=trialsPerDayLoaded)
    print(f"weights = {weights}")

    print(f"trimmed trials active: {trimmed_trials_active}")
    [print(f"{cat}: {all_conditions[cat].trials}") for cat in categories]

    for rep in (range(int(reps/K_fold_num))):

        #Need to shuffle only trimmed trials without perturbing where they lie in the the total set of trials
        shuffled_cachedTrainLogISIs = np.copy(cachedTrainLogISIs)
        shuffled_cachedTestLogISIs = np.copy(cachedTestLogISIs)
        perm = np.random.permutation(range(len(trimmed_trials_active)))
        shuffled_cachedTrainLogISIs[trimmed_trials_active] = shuffled_cachedTrainLogISIs[trimmed_trials_active][perm]
        shuffled_cachedTestLogISIs[trimmed_trials_active] = shuffled_cachedTestLogISIs[trimmed_trials_active][perm]

        folds = K_fold_strat(sessionfile,trimmed_trials_active,K_fold_num)
        for K, (Train_X, Test_X) in enumerate(folds):

            #if model is None:
            model = cachedtrainDecodingAlgorithm(sessionfile,clust,trialsPerDayLoaded,best_bw,cachedTrainLogISIs,Train_X,categories)
            fold_accuracy,fold_waccuracy,frac_empty = cachedcalculateAccuracyOnFold(sessionfile,clust,model,cachedTestLogISIs,Test_X,weights,conditions = categories)
            accuracy_per_fold.append(fold_accuracy)
            waccuracy_per_fold.append(fold_waccuracy)
            fraction_empty.append(frac_empty)
            
            #if model_c is None:
            model_c = cachedtrainDecodingAlgorithm(sessionfile,clust,trialsPerDayLoaded,best_bw,shuffled_cachedTrainLogISIs,Train_X,categories)
            cfold_accuracy,_,_ = cachedcalculateAccuracyOnFold(sessionfile,clust,model_c,shuffled_cachedTestLogISIs,Test_X,weights,conditions = categories)
            control_accuracy_per_fold.append(cfold_accuracy)

            #if model_s is None:
            model_s = cachedtrainDecodingAlgorithm(sessionfile,clust,trialsPerDayLoaded,best_bw,cachedTrainLogISIs,Train_X,categories,synthetic = True)
            sfold_accuracy,_,_ = cachedcalculateAccuracyOnFold(sessionfile,clust,model_s,cachedTestLogISIs,Test_X,weights,conditions = categories, synthetic = True)
            synthetic_accuracy_per_fold.append(sfold_accuracy)
            
    accuracy = np.nanmean(accuracy_per_fold)
    accuracy_std = np.nanstd(accuracy_per_fold)
    accuracy_sem = sem(accuracy_per_fold,nan_policy='omit')
    if type(accuracy_sem) == np.ma.core.MaskedConstant:
        accuracy_sem = np.nan
    
    waccuracy = np.nanmean(waccuracy_per_fold)
    waccuracy_std = np.nanstd(waccuracy_per_fold)
    waccuracy_sem = sem(waccuracy_per_fold,nan_policy='omit')
    if type(waccuracy_sem) == np.ma.core.MaskedConstant:
        waccuracy_sem = np.nan
    
    control_accuracy = np.nanmean(control_accuracy_per_fold)
    control_accuracy_std = np.nanstd(control_accuracy_per_fold)
    control_accuracy_sem = sem(control_accuracy_per_fold,nan_policy='omit')
    if type(control_accuracy_sem) == np.ma.core.MaskedConstant:
        control_accuracy_sem = np.nan

    synthetic_accuracy = np.nanmean(synthetic_accuracy_per_fold)
    synthetic_accuracy_std = np.nanstd(synthetic_accuracy_per_fold)
    synthetic_accuracy_sem = sem(synthetic_accuracy_per_fold,nan_policy='omit')
    if type(synthetic_accuracy_sem) == np.ma.core.MaskedConstant:
        synthetic_accuracy_sem = np.nan

    frac_empty = np.nanmean(fraction_empty)
    
    pval_c = mannwhitneyu(accuracy_per_fold,control_accuracy_per_fold).pvalue
    pval_s = mannwhitneyu(accuracy_per_fold,synthetic_accuracy_per_fold).pvalue

    results = dict()
    results['accuracy'] = accuracy
    results['accuracy_std'] = accuracy_std
    results['accuracy_sem'] = accuracy_sem
    results['weighted_accuracy'] = waccuracy
    results['weighted_accuracy_std'] = waccuracy_std
    results['weighted_accuracy_sem'] = waccuracy_sem
    results['shuffled_control_accuracy'] = control_accuracy
    results['shuffled_control_accuracy_std'] = control_accuracy_std
    results['shuffled_control_accuracy_sem'] = control_accuracy_sem
    results['synthetic_control_accuracy'] = synthetic_accuracy
    results['synthetic_control_accuracy_std'] = synthetic_accuracy_std
    results['synthetic_control_accuracy_sem'] = synthetic_accuracy_sem
    results['pval_shuffled_control'] = pval_c
    results['pval_synthetic_control'] = pval_s
    results['fraction_empty_trials'] = frac_empty
    print('regular results')
    print(results)
    return results

def generateNullResults():
    results = dict()
    results['accuracy'] = np.nan
    results['accuracy_std'] = np.nan
    results['accuracy_sem'] = np.nan
    results['weighted_accuracy'] = np.nan
    results['weighted_accuracy_std'] = np.nan
    results['weighted_accuracy_sem'] = np.nan
    results['shuffled_control_accuracy'] = np.nan
    results['shuffled_control_accuracy_std'] = np.nan
    results['shuffled_control_accuracy_sem'] = np.nan
    results['synthetic_control_accuracy'] = np.nan
    results['synthetic_control_accuracy_std'] = np.nan
    results['synthetic_control_accuracy_sem'] = np.nan
    results['pval_shuffled_control'] = np.nan
    results['pval_synthetic_control'] = np.nan
    results['fraction_empty_trials'] = np.nan
    print('null results')
    print(results)
    return results