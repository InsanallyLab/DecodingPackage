import numpy as np 
from itertools import product
from analysis import * 

class Session:
    def __init__(self, cluster, loader, trials_per_day_loaded):
        self.cluster = cluster
        self.trials_per_day_loaded = trials_per_day_loaded
        self.session_loader = loader 
        self.cached_log_ISIs = {}
        self.cached_times = None
        self.cached_conditions = {}
        self.cached_weights = None 

    def getSpikeTimes(self, starttime=np.nan,endtime=np.nan):
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
        if (not self.cached_times is None) and (not np.isnan(self.cluster)):
            print('ERROR: Cannot split cached spike times according to unit. Please amend query')
            raise Exception

        #If we've not cached a search already, we need to reset our cache to the total times
        if self.cached_times is None:
            self.cached_times = self.session_loader.spikes.times

        #Prune data by cluster. Remove spikes from all others
        if not np.isnan(self.cluster):
            clustidx = np.equal(self.session_loader.spikes.clusters,self.cluster)
            self.cached_times = self.cached_times[clustidx]

        #Remove spikes before starttime if applicable
        if not np.isnan(starttime):
            startidx = np.greater(self.cached_times,starttime)
            self.cached_times = self.cached_times[startidx]

        #Remove spikes after endtime if applicable
        if not np.isnan(endtime):
            endidx = np.less(self.cached_times,endtime)
            self.cached_times = self.cached_times[endidx]

        return self.cached_times

    def getLogISIs(self, interval, trials = None):
        """
        Computes and caches the log inter-spike intervals (LogISIs) for the given interval and trials.
        Args:
            interval (Interval): The time interval for which to compute the LogISIs.
            trials (list or None): A list of trial indices to consider. Default is None, which
                means all trials in the session will be considered.

        Returns:
            numpy.ndarray: An array containing LogISIs for each trial in the specified interval.

        Raises:
            ValueError: If `trials` contains invalid trial indices.
        """ 

        if (trials is None): 
            trials = range(self.loader.meta.length_in_trials) 

        # Check if LogISIs are already cached for the given interval and cluster
        cache_key = (trials, interval) 
        if cache_key in self.cached_log_ISIs:
            return self.cached_log_ISIs[cache_key]

        ISIs = []
        for trial in range(self.loader.meta.length_in_trials):
            starttime, endtime = interval._ToTimestamp(self.session_loader.trials, trial)
            spiketimes = self.getSpikeTimes(starttime=starttime, endtime=endtime)
            spiketimes = spiketimes * 1000 / self.loader.meta.fs

            ISIs.append(np.diff(spiketimes))

        LogISIs = np.array([np.log10(tISIs) for tISIs in ISIs], dtype='object')

        # Cache the LogISIs for future use
        self.cached_log_ISIs[cache_key] = LogISIs
        return LogISIs

    def getAllConditions(self, trialsPerDayLoaded=None): 
        """ 
        computes and caches conditions according to trialsPerDayLoaded 

        trialsPerDayLoaded: None or "NO_TRIM" 
        """
        if trialsPerDayLoaded is None: 
            trialsPerDayLoaded = self.trials_per_day_loaded 

        cache_key = trialsPerDayLoaded 

        if cache_key in self.cached_conditions: 
            return self.cached_conditions 

        #Some conditions currently disabled for the purposes of decoding
        trialOutcomes = ['NONE','target','nontarget','go','nogo','hit','miss','falarm','creject','slow_go','fast_go','correct','incorrect']
        laserConditions = ['NONE']
        switchConditions = ['NONE']

        if self.session_loader.meta.task == 'passive no behavior':
            trialOutcomes = ['NONE','target','nontarget']

        if self.session_loader.meta.task in ['opto nonreversal','opto switch','opto reversal']:
            laserConditions = ['NONE','laser_on','laser_off']
        if self.session_loader.meta.task in ['switch','opto switch','tuning switch']:
            switchConditions = ['NONE','pre_switch','post_switch']
        
        conditions = product(switchConditions,laserConditions,trialOutcomes)

        allconditions = dict()
        for cond in conditions:
            condition = getTrialSet(self.session_loader, self.cluster, cond,trialsPerDayLoaded=trialsPerDayLoaded)
            allconditions[condition.label] = condition

        self.cached_conditions[cache_key] = allconditions #cache conditions for future calls 

        return allconditions

    def splitByConditions(self, trials,conditions):
        all_conditions = self.getAllConditions(self.trials_per_day_loaded)
        decoding_conditions = dict()
        for cond in conditions:
            decoding_conditions[cond] = SimpleNamespace()
        
        for cond in decoding_conditions:
            condition_trials = trials[np.isin(trials, all_conditions[cond].trials )]
            decoding_conditions[cond].trials = condition_trials
            
        return decoding_conditions

    def calculateWeights(self, trimmed_trials_active,conditions):
        if self.cached_weights is not None: 
            return self.cached_weights 
        
        num_total_trials = len(trimmed_trials_active)
        all_conditions = self.getAllConditions(self.trials_per_day_loaded)

        weights = dict()
        for cat in conditions:
            weights[cat] = len(all_conditions[cat].trials)/num_total_trials
            weights[cat] = (1 / weights[cat]) / len(conditions)

        self.cached_weights = weights 
    
        return weights
