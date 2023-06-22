import os
import pickle 
import numpy as np 
from itertools import product 
from types import SimpleNamespace


#from io.py 
def loadSessionCached(destination,filename):
    with open(os.path.join(destination,filename), 'rb') as f:
        session = pickle.load(f)
    return session

##from utility.py 
def generateDateString(sessionfile):
    namemodifier = 'ERROR'
    if sessionfile.meta.task == 'CNO':
        namemodifier = 'CNO'
    elif sessionfile.meta.task in ['nonreversal','switch','reversal','second switch','second reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)
    elif sessionfile.meta.task in ['opto nonreversal','opto switch','opto reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)
    elif sessionfile.meta.task in ['tuning nonreversal','tuning switch','tuning reversal']:
        namemodifier = str(sessionfile.meta.day_of_recording)
    elif sessionfile.meta.task == 'thalamus tuning':
        namemodifier = str(sessionfile.meta.day_of_recording)

    return sessionfile.meta.animal + '_' + namemodifier + '_' + sessionfile.meta.region + '_' + str(sessionfile.meta.date).replace('/','-')

###### Out-ouf-File Dependencies ############################################## 

def getSpikeTimes(sessionfile,clust=np.nan,starttime=np.nan,endtime=np.nan,cachedtimes=None):
    """
    set clust to control what neuron id to search for
    set starttime and endtime (in samples) to control time span
    if not set, starttime and endtime will each default to the start
    and end of the recording
    pass cached searches into cachedtimes to speed up sequential reads
    from the same neuron's spike times

    returns spike times relative to start of recording in samples
    """

    #Since we don't actually return any spike cluster identity, we cannot cache it
    #and thus cannot split any cached data according to cluster identity. This could
    #be amended but it is not currently intended behavior to allow this operation.
    if (not cachedtimes is None) and (not np.isnan(clust)):
        print('ERROR: Cannot split cached spike times according to unit. Please amend query')
        raise Exception

    #If we've not cached a search already, we need to reset our cache to the total times
    if cachedtimes is None:
        cachedtimes = sessionfile.spikes.times

    #Prune data by cluster. Remove spikes from all others
    if not np.isnan(clust):
        clustidx = np.equal(sessionfile.spikes.clusters,clust)
        cachedtimes = cachedtimes[clustidx]

    #Remove spikes before starttime if applicable
    if not np.isnan(starttime):
        startidx = np.greater(cachedtimes,starttime)
        cachedtimes = cachedtimes[startidx]

    #Remove spikes after endtime if applicable 
    if not np.isnan(endtime):
        endidx = np.less(cachedtimes,endtime)
        cachedtimes = cachedtimes[endidx]

    return cachedtimes

def getTrialSpikes(sessionfile,trial,clust=np.nan,cachedtimes=None,startbuffer=0,endbuffer=0,outunits='samples'):
    """
    set trial to control time span (0-indexed)
    set clust to control what neuron id to search for
    pass cached searches from getSpikeTimes into cachedtimes
    to speed up sequential reads from the same neuron's
    spike times
    set startbuffer and endbuffer to buffer data at the start or end of a trial

    returns spike times relative to trial start in units of outunits
    """

    startbuffer *= sessionfile.meta.fs
    endbuffer *= sessionfile.meta.fs

    trialstart = sessionfile.trials.starts[trial]
    trialend = sessionfile.trials.ends[trial]

    times = np.array(getSpikeTimes(sessionfile,clust=clust,starttime=(trialstart-startbuffer),endtime=(trialend+endbuffer),cachedtimes=cachedtimes),dtype='float')
    times -= trialstart

    if outunits in ['ms','milliseconds']:
        times = times / sessionfile.meta.fs * 1000
    elif outunits in ['s','seconds']:
        times = times / sessionfile.meta.fs
    elif outunits=='samples':
        pass

    return times


##from analysis.py 

def getTrialSet(sessionfile,clust,listOfTrialSets,trialsPerDayLoaded):
    trimmed_trials_active = np.array(sessionfile.trim[clust].trimmed_trials)
    if trialsPerDayLoaded == 'NO_TRIM':
        pass
    else:
        active_trials = trialsPerDayLoaded[sessionfile.meta.animal][sessionfile.meta.day_of_training]
        trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)]

    trials = np.copy(trimmed_trials_active)
    label = ''
    for trialSet in listOfTrialSets:
        if trialSet == 'NONE':
            continue
        if trialSet == 'target':
            trialsTrim = np.array(np.where(sessionfile.trials.target)[0])
        elif trialSet == 'nontarget':
            trialsTrim = np.array(np.where(np.logical_not(sessionfile.trials.target))[0])
        elif trialSet == 'go':
            trialsTrim = np.array(np.where(sessionfile.trials.go)[0])
        elif trialSet == 'nogo':
            trialsTrim = np.array(np.where(np.logical_not(sessionfile.trials.go))[0])
        elif trialSet == 'hit':
            trialsTrim = np.array(np.where(np.logical_and(sessionfile.trials.target,sessionfile.trials.go))[0])
        elif trialSet == 'miss':
            trialsTrim = np.array(np.where(np.logical_and(sessionfile.trials.target,np.logical_not(sessionfile.trials.go)))[0])
        elif trialSet == 'falarm':
            trialsTrim = np.array(np.where(np.logical_and(np.logical_not(sessionfile.trials.target),sessionfile.trials.go))[0])
        elif trialSet == 'creject':
            trialsTrim = np.array(np.where(np.logical_and(np.logical_not(sessionfile.trials.target),np.logical_not(sessionfile.trials.go)))[0])
        elif trialSet == 'slow_go':
            response_times = np.array(sessionfile.trials.response) - np.array(sessionfile.trials.starts)
            response_times_go = response_times[sessionfile.trials.go]
            mean_response_time = np.mean(response_times_go)
            slow_go_threshold = max(2*mean_response_time,0.5*sessionfile.meta.fs)
            slow_go_response = np.where(np.greater(response_times,slow_go_threshold))[0]
            trialsTrim = slow_go_response
        elif trialSet == 'fast_go':
            response_times = np.array(sessionfile.trials.response) - np.array(sessionfile.trials.starts)
            response_times_go = response_times[sessionfile.trials.go]
            mean_response_time = np.mean(response_times_go)
            fast_go_threshold = max(2*mean_response_time,0.5*sessionfile.meta.fs)
            fast_go_response = np.where(np.less(response_times,fast_go_threshold))[0]
            fast_go_response = fast_go_response[np.isin(fast_go_response,trimmed_trials_active)]
            trialsTrim = fast_go_response
        elif trialSet == 'correct':
            trialsHit = np.array(np.where(np.logical_and(sessionfile.trials.target,sessionfile.trials.go))[0])
            trialsCrej = np.array(np.where(np.logical_and(np.logical_not(sessionfile.trials.target),np.logical_not(sessionfile.trials.go)))[0])
            trialsTrim = np.unique(np.concatenate((trialsHit,trialsCrej)))
        elif trialSet == 'incorrect':
            trialsMiss = np.array(np.where(np.logical_and(sessionfile.trials.target,np.logical_not(sessionfile.trials.go)))[0])
            trialsFal = np.array(np.where(np.logical_and(np.logical_not(sessionfile.trials.target),sessionfile.trials.go))[0])
            trialsTrim = np.unique(np.concatenate((trialsMiss,trialsFal)))
        elif trialSet == 'laser_on':
            trialsTrim = np.array(np.where(sessionfile.trials.laser_stimulation)[0])
        elif trialSet == 'laser_off':
            trialsTrim = np.array(np.where(np.logical_not(sessionfile.trials.laser_stimulation))[0])
        elif trialSet == 'pre_switch':
            trialsTrim = np.array(range(sessionfile.meta.first_reversal_trial))
        elif trialSet == 'post_switch':
            trialsTrim = np.array(range(sessionfile.meta.first_reversal_trial,sessionfile.meta.length_in_trials))
        else:
            raise Exception('Unrecognized Trial Set')
        trials = trials[np.isin(trials,trialsTrim)]

        if label == '':
            label = trialSet
        else:
            label = label + '_' + trialSet
    if label == '':
        label = 'all_trials'

    condition = SimpleNamespace()
    condition.trials = trials
    condition.label = label
    return condition

def getAllConditions(sessionfile,clust,trialsPerDayLoaded=None):
    #Some conditions currently disabled for the purposes of decoding
    trialOutcomes = ['NONE','target','nontarget','go','nogo','hit','miss','falarm','creject','slow_go','fast_go','correct','incorrect']
    laserConditions = ['NONE']
    switchConditions = ['NONE']

    if sessionfile.meta.task == 'passive no behavior':
        trialOutcomes = ['NONE','target','nontarget']

    if sessionfile.meta.task in ['opto nonreversal','opto switch','opto reversal']:
        laserConditions = ['NONE','laser_on','laser_off']
    if sessionfile.meta.task in ['switch','opto switch','tuning switch']:
        switchConditions = ['NONE','pre_switch','post_switch']
    
    conditions = product(switchConditions,laserConditions,trialOutcomes)

    allconditions = dict()
    for cond in conditions:
        condition = getTrialSet(sessionfile,clust,cond,trialsPerDayLoaded=trialsPerDayLoaded)
        allconditions[condition.label] = condition

    return allconditions

