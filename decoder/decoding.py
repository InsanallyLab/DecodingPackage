
import numpy as np 
from .bandwidth import sklearn_grid_search_bw
from scipy.stats import sem, mannwhitneyu 
from sklearn.model_selection import StratifiedKFold
from .utility import *
from .analysis import * 
from .training import * 



CAT_TO_COND = {  
    'stimulus':['target','nontarget'], 'response': ['go','nogo'],  'stimulus_off': ['laser_off_target','laser_off_nontarget'], 
    'stimulus_on':['laser_on_target','laser_on_nontarget'], 'response_off': ['laser_off_go','laser_off_nogo'], 'response_on': ['laser_on_go','laser_on_nogo'] 
} 

class DecodingAlgorithm: 

    def __init__(self, loader, clust, trialsPerDayLoaded, trainInterval, testInterval, reps, categories): 

        self.loader = loader 
        self.model = None 
        self.clust =  clust
        self.trialsPerDayLoaded = trialsPerDayLoaded 
        self.trainInterval = trainInterval 
        self.testInterval = testInterval 
        self.reps = reps
        self.categories = categories

    def cacheLogISIs(self, interval):#Include conditions
        ISIs = []
        for trial in range(self.loader.meta.length_in_trials):
            starttime,endtime = interval._ToTimestamp(self.loader.trials,trial)
            spiketimes = getSpikeTimes(self.loader.spikes,self.clust,starttime,endtime)
            spiketimes = spiketimes * 1000 / self.loader.meta.fs

            ISIs.append(np.diff(spiketimes))

        LogISIs = np.array([np.log10(tISIs) for tISIs in ISIs],dtype='object')
        return LogISIs
    
    def calculate_weights(self, trimmed_trials_active):
        num_total_trials = len(trimmed_trials_active)
        all_conditions = getAllConditions(self.loader,self.clust,self.trialsPerDayLoaded)

        weights = dict()
        for cat in self.categories:
            weights[cat] = len(all_conditions[cat].trials)/num_total_trials
            weights[cat] = (1 / weights[cat]) / len(self.categories)
        return weights
    
    def K_fold_strat(self, trials,K):
        trials = np.array(trials)
        
        X = np.ones(len(trials))
        y = np.ones(len(trials))
        for idx,trial in enumerate(trials):
            if self.loader.trials["target"][trial] and self.loader.trials["go"][trial]:
                y[idx] = 1
            elif self.loader.trials["target"][trial] and not self.loader.trials["go"][trial]:
                y[idx] = 2
            elif not self.loader.trials["target"][trial] and self.loader.trials["go"][trial]:
                y[idx] = 3
            elif not self.loader.trials["target"][trial] and not self.loader.trials["go"][trial]:
                y[idx] = 4
        
        train_test_pairs = []
        skf = StratifiedKFold(n_splits=K,shuffle=True)
        for splitX,splitY in skf.split(X, y):
            train_trials = trials[splitX]
            test_trials = trials[splitY]
            train_test_pairs.append((train_trials,test_trials))
        return train_test_pairs

    def cachedCalculateClusterAccuracy(self, bw = None, K_fold_num = 10): 
        if bw is None:
            best_bw = sklearn_grid_search_bw(self.loader,self.clust,self.trialsPerDayLoaded,self.trainInterval)
        else:
            best_bw = bw
            
        accuracy_per_fold = []
        waccuracy_per_fold = []
        
        control_accuracy_per_fold = []
        
        synthetic_accuracy_per_fold = []

        fraction_empty = []
        self.categories = CAT_TO_COND[self.categories]
        categories = self.categories
        cachedTrainLogISIs = self.cacheLogISIs(self.trainInterval)
        cachedTestLogISIs = self.cacheLogISIs(self.testInterval)

        model = None
        model_c = None #Condition permutation
        model_s = None #Synthetic spiketrains

        #Get active trimmed trials
        trimmed_trials_active = np.array(self.loader.trimmed_trials[self.clust].trimmed_trials)
        if self.trialsPerDayLoaded != 'NO_TRIM':
            active_trials = self.trialsPerDayLoaded[self.loader.meta.animal][self.laoder.meta.day_of_training]
            trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)]

        #Remove all trials that do not belong to one of the conditions in question
        included_in_conditions_mask = []
        all_conditions = getAllConditions(self.loader, self.clust,self.trialsPerDayLoaded)
        for category in categories:
            included_in_conditions_mask = np.concatenate((included_in_conditions_mask,all_conditions[category].trials))
        trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,included_in_conditions_mask)]

        weights = self.calculate_weights(trimmed_trials_active)
        print(f"weights = {weights}")

        print(f"trimmed trials active: {trimmed_trials_active}")
        [print(f"{cat}: {all_conditions[cat].trials}") for cat in categories]

        for rep in (range(int(self.reps/K_fold_num))):

            #Need to shuffle only trimmed trials without perturbing where they lie in the the total set of trials
            shuffled_cachedTrainLogISIs = np.copy(cachedTrainLogISIs)
            shuffled_cachedTestLogISIs = np.copy(cachedTestLogISIs)
            perm = np.random.permutation(range(len(trimmed_trials_active)))
            shuffled_cachedTrainLogISIs[trimmed_trials_active] = shuffled_cachedTrainLogISIs[trimmed_trials_active][perm]
            shuffled_cachedTestLogISIs[trimmed_trials_active] = shuffled_cachedTestLogISIs[trimmed_trials_active][perm]

            folds = self.K_fold_strat(trimmed_trials_active,K_fold_num)
            for K, (Train_X, Test_X) in enumerate(folds):

                #if model is None:
                model = cachedtrainDecodingAlgorithm(self.loader,self.clust,self.trialsPerDayLoaded,best_bw,cachedTrainLogISIs,Train_X,categories)
                fold_accuracy,fold_waccuracy,frac_empty = cachedcalculateAccuracyOnFold(self.loader,self.clust,model,cachedTestLogISIs,Test_X,weights,conditions = categories)
                accuracy_per_fold.append(fold_accuracy)
                waccuracy_per_fold.append(fold_waccuracy)
                fraction_empty.append(frac_empty)
                
                #if model_c is None:
                model_c = cachedtrainDecodingAlgorithm(self.loader,self.clust,self.trialsPerDayLoaded,best_bw,shuffled_cachedTrainLogISIs,Train_X,categories)
                cfold_accuracy,_,_ = cachedcalculateAccuracyOnFold(self.loader,self.clust,model_c,shuffled_cachedTestLogISIs,Test_X,weights,conditions = categories)
                control_accuracy_per_fold.append(cfold_accuracy)

                #if model_s is None:
                model_s = cachedtrainDecodingAlgorithm(self.loader,self.clust,self.trialsPerDayLoaded,best_bw,cachedTrainLogISIs,Train_X,categories,synthetic = True)
                sfold_accuracy,_,_ = cachedcalculateAccuracyOnFold(self.loader,self.clust,model_s,cachedTestLogISIs,Test_X,weights,conditions = categories, synthetic = True)
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

        


