import numpy as np
import pickle
import os
from types import SimpleNamespace
import util 
from interval import TrialInterval  
import training 
from io import * 
from decoding import DecodingAlgorithm

class Decoder: 
    def __init__(self, loader): 
        """ 
        params: loader: the result from load_session(data)
        """
        self.loader = loader #for now this is a sessionfilee loader from load.py 

    def synthetic_spiketrain(self,trial_length=2500):#only model.all is required to be filled out. Units in ms
        LogISIs = []
        ctime = 0
        while True:
            ISI = self.model.all.Inv_Likelihood(np.random.rand())
            ctime += 10**ISI
            if ctime <= trial_length:
                LogISIs.append(ISI)
            else:
                break
        return np.array(LogISIs)
    
    def cachedpredictTrial(self,model,trialISIs,conditions = ['target_tone','nontarget_tone'],synthetic = False):
        if synthetic:
            LogISIs = self.synthetic_spiketrain()
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

    def generateNullResults(self):
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

    
        
    def calculateDecodingForSingleNeuron(self,clust,trialsPerDayLoaded,output_directory,trainInterval,testInterval,reps = 1,categories='stimulus'): 

        filename = util.generateDateString(self.loader.meta) + ' cluster ' + str(clust) + ' decoding cached result.pickle'
        filename = os.path.join(output_directory,filename)

        decoder = DecodingAlgorithm(self.loader, clust, trialsPerDayLoaded, trainInterval, testInterval, reps, categories)
        
        try:    
            trainInterval._CalculateAvgLickDelay(self.loader.trials)
            testInterval._CalculateAvgLickDelay(self.loader.trials)
            res = decoder.cachedCalculateClusterAccuracy()
        except Exception as e:
            res = self.generateNullResults()
            print(f"session cluster {clust} has thrown error {e}")
            raise e
        try:
            with open(filename, 'wb') as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Problem saving {f} to {filename}. Error: {e}")
                
        print(f"finished with {util.generateDateString(self.loader.meta)} cluster {clust}")
        return res

    ################################### Validation ################################
    
    def Train_Test_Split(trials,frac_test = 0.1):
        """
        Splits a set of trials into test and train datasets
        trials: set of trials available in dataset
        frac_test: fraction of trails to use for test (0 for leave-one-out)
        returns (train_trials,test_trials)
        """
        
        N = len(trials)
        
        #Test set size. Must be at least one
        N_test = int(frac_test * N)
        if N_test < 1:
            N_test = 1
            
        #Train set size. Must also be at least one.
        N_train = N - N_test
        if N_train < 1:
            print('ERROR: No training data. Test fraction likely too high')
            raise Exception
            
        test_idxs = np.concatenate(( [False]*N_train , [True]*N_test ))
        test_idxs = np.random.permutation(test_idxs)
        train_idxs = np.logical_not(test_idxs)
        
        test_trials = trials[test_idxs]
        train_trials = trials[train_idxs]
        
        return (train_trials,test_trials)


    











