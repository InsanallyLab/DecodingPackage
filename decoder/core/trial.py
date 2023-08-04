
from types import SimpleNamespace 
import numpy as np 
from  itertools import product 

class Trial: 
    def __init__(self, loader, cluster, trials_per_day_loaded) -> None:
        self.loader = loader
        self.cluster = cluster
        self.trials_per_day_loaded = trials_per_day_loaded
        self.cached_conditions = {}
        self.cached_weights = None 

    def _getTrialSet(self, listOfTrialSets,trialsPerDayLoaded):
        clust = self.cluster 
        loader = self.loader 
        trimmed_trials_active = self.loader.trimmed_trials(self.cluster)
        if trialsPerDayLoaded == 'NO_TRIM':
            pass
        else:
            active_trials = trialsPerDayLoaded[self.loader.meta.animal][self.loader.meta.day_of_training]
            trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)]

        trials = np.copy(trimmed_trials_active)
        label = ''
        for trialSet in listOfTrialSets:
            if trialSet == 'NONE':
                continue
            if trialSet == 'target':
                trialsTrim = np.array(np.where(self.loader.trials["target"])[0])
            elif trialSet == 'nontarget':
                trialsTrim = np.array(np.where(np.logical_not(self.loader.trials["target"]))[0])
            elif trialSet == 'go':
                trialsTrim = np.array(np.where(self.loader.trials["go"])[0])
            elif trialSet == 'nogo':
                trialsTrim = np.array(np.where(np.logical_not(self.loader.trials["go"]))[0])
            elif trialSet == 'hit':
                trialsTrim = np.array(np.where(np.logical_and(self.loader.trials["target"],self.loader.trials["go"]))[0])
            elif trialSet == 'miss':
                trialsTrim = np.array(np.where(np.logical_and(self.loader.trials["target"],np.logical_not(self.loader.trials["go"])))[0])
            elif trialSet == 'falarm':
                trialsTrim = np.array(np.where(np.logical_and(np.logical_not(self.loader.trials["target"]),self.loader.trials["go"]))[0])
            elif trialSet == 'creject':
                trialsTrim = np.array(np.where(np.logical_and(np.logical_not(self.loader.trials["target"]),np.logical_not(self.loader.trials["go"])))[0])
            elif trialSet == 'slow_go':
                response_times = loader.trials["response"] - loader.trials["starts"]
                response_times_go = response_times[loader.trials["go"]]
                mean_response_time = np.mean(response_times_go)
                slow_go_threshold = max(2*mean_response_time,0.5*loader.meta.fs)
                slow_go_response = np.where(np.greater(response_times,slow_go_threshold))[0]
                trialsTrim = slow_go_response
            elif trialSet == 'fast_go':
                response_times = loader.trials["response"] - loader.trials["starts"]
                response_times_go = response_times[loader.trials["go"]]
                mean_response_time = np.mean(response_times_go)
                fast_go_threshold = max(2*mean_response_time,0.5*loader.meta.fs)
                fast_go_response = np.where(np.less(response_times,fast_go_threshold))[0]
                fast_go_response = fast_go_response[np.isin(fast_go_response,trimmed_trials_active)]
                trialsTrim = fast_go_response
            elif trialSet == 'correct':
                trialsHit = np.array(np.where(np.logical_and(loader.trials["target"],loader.trials["go"]))[0])
                trialsCrej = np.array(np.where(np.logical_and(np.logical_not(loader.trials["target"]),np.logical_not(loader.trials["go"])))[0])
                trialsTrim = np.unique(np.concatenate((trialsHit,trialsCrej)))
            elif trialSet == 'incorrect':
                trialsMiss = np.array(np.where(np.logical_and(loader.trials["target"],np.logical_not(loader.trials["go"])))[0])
                trialsFal = np.array(np.where(np.logical_and(np.logical_not(loader.trials["target"]),loader.trials["go"]))[0])
                trialsTrim = np.unique(np.concatenate((trialsMiss,trialsFal)))
            elif trialSet == 'laser_on':
                trialsTrim = np.array(np.where(loader.trials["laser_stimulation"])[0])
            elif trialSet == 'laser_off':
                trialsTrim = np.array(np.where(np.logical_not(loader.trials["laser_stimulation"]))[0])
            elif trialSet == 'pre_switch':
                trialsTrim = np.array(range(loader.meta.first_reversal_trial))
            elif trialSet == 'post_switch':
                trialsTrim = np.array(range(loader.meta.first_reversal_trial,loader.meta.length_in_trials))
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

    def get_all_conditions(self, trialsPerDayLoaded=None): 
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

        if self.loader.meta.task == 'passive no behavior':
            trialOutcomes = ['NONE','target','nontarget']

        if self.loader.meta.task in ['opto nonreversal','opto switch','opto reversal']:
            laserConditions = ['NONE','laser_on','laser_off']
        if self.loader.meta.task in ['switch','opto switch','tuning switch']:
            switchConditions = ['NONE','pre_switch','post_switch']
        
        conditions = product(switchConditions,laserConditions,trialOutcomes)

        allconditions = dict()
        for cond in conditions:
            condition = self._getTrialSet(cond,trialsPerDayLoaded=trialsPerDayLoaded)
            allconditions[condition.label] = condition

        self.cached_conditions[cache_key] = allconditions #cache conditions for future calls 

        return allconditions

    def split_by_conditions(self, trials,conditions):
        all_conditions = self.get_all_conditions(self.trials_per_day_loaded)
        decoding_conditions = dict()
        for cond in conditions:
            decoding_conditions[cond] = SimpleNamespace()
        
        for cond in decoding_conditions:
            condition_trials = trials[np.isin(trials, all_conditions[cond].trials )]
            decoding_conditions[cond].trials = condition_trials
            
        return decoding_conditions

    def calculate_weights(self, conditions, num_trials):
        if self.cached_weights is not None: 
            return self.cached_weights 
        
        num_total_trials = num_trials 
        all_conditions = self.get_all_conditions(self.trials_per_day_loaded)

        weights = dict()
        for cat in conditions:
            weights[cat] = len(all_conditions[cat].trials)/num_total_trials
            weights[cat] = (1 / weights[cat]) / len(conditions)

        self.cached_weights = weights 
    
        return weights
