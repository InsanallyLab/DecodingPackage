from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_X_y
from scipy.stats import sem, mannwhitneyu 
from types import SimpleNamespace  
from session import Session
from spiketrain import SpikeTrain 

import numpy as np 
import pandas as pd 

CAT_TO_COND = {  
    'stimulus':['target','nontarget'], 'response': ['go','nogo'],  'stimulus_off': ['laser_off_target','laser_off_nontarget'], 
    'stimulus_on':['laser_on_target','laser_on_nontarget'], 'response_off': ['laser_off_go','laser_off_nogo'], 'response_on': ['laser_on_go','laser_on_nogo'] 
} 

class NeuralDecoder(BaseEstimator, ClassifierMixin):
    def __init__(self, trialsperdayloaded, cluster, loader, categories, conditions):
        #params initialized in 
        self.loader = loader
        self.cluster =  cluster
        self.trialsPerDayLoaded = trialsperdayloaded 
        self.trainInterval = None
        self.testInterval = None 
        self.reps = None
        self.categories = categories
        self.conditions = conditions 
        self.session = Session(cluster, loader, trialsperdayloaded)
        self.spike_train = SpikeTrain(self.session)

    def get_active_trimmed_trials(self): 
        #Get active trimmed trials
        #trimmed_trials_active = np.array(sessionfile.trim[clust].trimmed_trials) 
        trimmed_trials_active = self.loader.trimmed_trials_active(self.cluster) #TODO: make this method in loader class

        if self.trialsPerDayLoaded != 'NO_TRIM':
            active_trials = self.trialsPerDayLoaded[self.loader.meta.animal][self.loader.meta.day_of_training]
            trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,active_trials)]

        #Remove all trials that do not belong to one of the conditions in question
        included_in_conditions_mask = []
        all_conditions = self.session.getAllConditions(trialsPerDayLoaded=self.trialsPerDayLoaded)
        for category in self.conditions:
            included_in_conditions_mask = np.concatenate((included_in_conditions_mask,all_conditions[category].trials))
        trimmed_trials_active = trimmed_trials_active[np.isin(trimmed_trials_active,included_in_conditions_mask)]

        return trimmed_trials_active 

    def k_fold_strat(self, trials,  n_splits): 
        K = n_splits 
        trials = np.array(trials)
    
        X = np.ones(len(trials))
        y = np.ones(len(trials))

        #TODO: hard coded labels for all four types of trials 
        for idx,trial in enumerate(trials):
            if self.loader.trials.target[trial] and self.loader.trials.go[trial]:
                y[idx] = 1
            elif self.loader.trials.target[trial] and not self.loader.trials.go[trial]:
                y[idx] = 2
            elif not self.loader.trials.target[trial] and self.loader.trials.go[trial]:
                y[idx] = 3
            elif not self.loader.trials.target[trial] and not self.loader.trials.go[trial]:
                y[idx] = 4
        
        train_test_pairs = []
        skf = StratifiedKFold(n_splits=K,shuffle=True)
        for splitX,splitY in skf.split(X, y):        
            train_trials = trials[splitX]
            test_trials = trials[splitY]
            train_test_pairs.append((train_trials,test_trials))
        return train_test_pairs

    def fit(self, train_interval, train_x, bw = None, synthetic = False): 
        """ 
        
        returns model 
        """
        cachedLogISIs = self.session.getLogISIs(train_interval) 

        model = SimpleNamespace()
        model.conds = dict()
        model.all = SimpleNamespace()
        for cond in self.conditions:
            model.conds[cond] = SimpleNamespace()
        
        #Determine trials to use for each condition. Uses the conditions structures from
        #ilep to increase modularity and to ensure that all code is always using the same
        #conditions
        decoding_conditions = splitByConditions(sessionfile,clust,trialsPerDayLoaded,Train_X,condition_names)
        
        LogISIs = cachedLogISIs[train_x]
        if len(LogISIs)<5:
            return None
        LogISIs = np.concatenate(LogISIs)
        if len(LogISIs)<5:
            return None

        f,inv_f = SpikeTrain.estimate_isi_distribution(LogISIs, bw)
        model.all.Likelihood = f
        model.all.Inv_Likelihood = inv_f

        #Handle synthetic spiketrain generation
        if synthetic:
            synthetic_spiketrain_construction_set = np.copy(cachedLogISIs)
            cachedLogISIs = [ [] ]*self.loader.meta.length_in_trials
            for trial in train_x:
                cachedLogISIs[trial] = SpikeTrain.synthetic_spiketrain(synthetic_spiketrain_construction_set[train_x])
            cachedLogISIs = np.array(cachedLogISIs,dtype='object')

        #Handle individual conditions
        LogISIs_per_trial = [len(l) for l in cachedLogISIs[train_x]]
        total_empty_ISI_trials = np.sum(np.equal(  LogISIs_per_trial,0  ))
        for cond in decoding_conditions:        
            LogISIs = cachedLogISIs[decoding_conditions[cond].trials]
            LogISIs = np.concatenate(LogISIs)
            if len(LogISIs)<5:
                print(f"Skipping fold. Not enough ISIs for the {cond} condition")
                return None
            f,inv_f = SpikeTrain.estimate_isi_distribution(LogISIs, bw) #This is a gaussian KDE
            model.conds[cond].Likelihood = f
            model.conds[cond].Inv_Likelihood = inv_f
            model.conds[cond].Prior_0 = 1.0 / len(self.conditions)

            #Calculate likelihood for 0-ISI trials
            LogISIs_per_trial = [len(l) for l in cachedLogISIs[decoding_conditions[cond].trials]]
            numerator = np.sum(np.equal(LogISIs_per_trial,0)) + 1
            denominator = total_empty_ISI_trials + len(self.conditions)
            model.conds[cond].Prior_empty = numerator / denominator
            
        return model

    def decode(self, n_reps, trimmed_trials_active, K_fold_num, synthetic = False): 
        """ 
        performs decoding n_reps times with k-fold cross validation 

        returns 
        """ 
        accuracy_per_fold = []
        waccuracy_per_fold = []

        synthetic_accuracy_per_fold = []
        synthetic_waccuracy_per_fold = [] 

        fraction_empty = []
        model = None
        model_s = None #Synthetic spiketrains

        for _ in (range(int(n_reps/K_fold_num))):

            folds = self.k_fold_strat(trimmed_trials_active,K_fold_num)
            for _, (Train_X, Test_X) in enumerate(folds):

                #actual model 
                model = fit(sessionfile,clust,trialsPerDayLoaded,best_bw,cachedTrainLogISIs,Train_X,categories)
                if model is None:
                    fold_accuracy,fold_waccuracy,frac_empty = (np.nan,np.nan,np.nan)
                else:
                    fold_accuracy,fold_waccuracy,frac_empty = self.decode_single(trimmed_trials_active, model, Test_X)
                    print(f"fold accuracy is {fold_accuracy}. w = {fold_waccuracy}")
                accuracy_per_fold.append(fold_accuracy)
                waccuracy_per_fold.append(fold_waccuracy)
                fraction_empty.append(frac_empty)

                #synthetic spiketrain model 
                model_s = cachedtrainDecodingAlgorithm(sessionfile,clust,trialsPerDayLoaded,best_bw,cachedTrainLogISIs,Train_X,categories,synthetic = True)
                if model_s is None:
                    sfold_accuracy = np.nan
                else:
                    sfold_accuracy,sfold_waccuracy,_ = self.decode_single(trimmed_trials_active, model_s, Test_X)
                synthetic_accuracy_per_fold.append(sfold_accuracy)
                synthetic_waccuracy_per_fold.append(sfold_waccuracy)


        results = {"accuracy": accuracy_per_fold, "weighted_accuracy" : waccuracy_per_fold,
        "synthetic_accuracy": synthetic_accuracy_per_fold, "synthetic_weighted_accuracy": synthetic_waccuracy_per_fold, "fraction_empty": frac_empty} 

        return pd.dataFrame(results)

    

    def decode_single(self, trimmed_trials_active, model, test_x, synthetic = False): 
        """ 
        returns 
            accuracy, weighted accuracy, fraction empty 
        """
        weights = self.session.calculateWeights(trimmed_trials_active, self.conditions) 
        
        testlogISIs = self.session.getLogISIs(self.testInterval) 
        #Note: this call to getAllConditions uses NO_TRIM all the time because it is only used to determine correctness
        all_conditions = self.session.getAllConditions(trialsPerDayLoaded='NO_TRIM')
        
        accumulated_correct = 0
        #accumulated_total = 0 # I have changed how weighting works to be weighted by condition prevalence so this is no longer necessary
        num_correct = 0
        num_total = 0
        num_empty = 0
        
        for trial in test_x:
            synthetic_spiketrain_construction_set = np.copy(testlogISIs)[test_x]
            cond,prob,_,empty_ISIs = self.predict_single_trial(model,testlogISIs[trial],synthetic=synthetic,synthetic_spiketrain_construction_set=synthetic_spiketrain_construction_set)
            
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

    def predict_single_trial(self, model, trialISIs, synthetic, synthetic_spiketrain_construction_set): 
        """ 
        returns 
            predicted condition, probability of predicted cond, probabilities, is_empty_isi 
        """
        if synthetic:
            LogISIs = SpikeTrain.synthetic_spiketrain(synthetic_spiketrain_construction_set)
        else:
            LogISIs = trialISIs
        
        #Set up probabilities with initial values equal to priors
        probabilities = dict()
        for cond in self.conditions:
            probabilities[cond] = SimpleNamespace()
            probabilities[cond].prob = np.full(len(LogISIs)+1,np.nan)
            probabilities[cond].prob[0] =  np.log10(model.conds[cond].Prior_0)
        
        #Calculate change in W. LLR after each LogISI
        for cond in self.conditions:
            probabilities[cond].prob = np.cumsum(np.log10(np.concatenate((   [model.conds[cond].Prior_0] , model.conds[cond].Likelihood(LogISIs)   ))))

        #Exponentiate back from log so that normalization can take place
        for cond in self.conditions:
            probabilities[cond].prob = np.power(10,probabilities[cond].prob)
        
        #Calculate total probability sum to normalize against
        sum_of_probs = np.zeros(len(LogISIs)+1)  
        for cond in self.conditions:
            sum_of_probs += probabilities[cond].prob

        #Normalize all probabilities
        for cond in self.conditions:
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
            return maxCond,probs[maxidx],probabilities,True
        #ISIs were present in the trial. We can take out a proper choice by the algorithm.
        else:
            keys = [cond for cond in probabilities]
            probs = [probabilities[cond].prob for cond in probabilities]
            probs = [p[len(p)-1] for p in probs]
            maxidx = np.argmax(probs)
            maxCond = keys[maxidx]
            return maxCond,probs[maxidx], probabilities,False


    def evaluate(): 
        return 

