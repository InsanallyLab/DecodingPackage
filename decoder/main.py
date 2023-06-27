import numpy as np
import pickle
import os
from types import SimpleNamespace
import util 
from interval import TrialInterval  
import training 
import results
from io import * 

class Decoder: 
    def __init__(self, loader): 
        """ 
        params: loader: the result from load_session(data)
        """
        self.model = SimpleNamespace()
        self.data = loader #for now this is a sessionfilee loader from load.py 

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
    
    def cachedtrainDecodingAlgorithm(self, sessionfile,clust,trialsPerDayLoaded,bw,cachedLogISIs,Train_X,condition_names,synthetic = False):
        self.model.conds = dict()
        self.model.all = SimpleNamespace()
        for cond in condition_names:
            self.model.conds[cond] = SimpleNamespace()
        
        #Determine trials to use for each condition. Uses the conditions structures from
        #ilep to increase modularity and to ensure that all code is always using the same
        #conditions
        decoding_conditions = training.splitByConditions(sessionfile,clust,trialsPerDayLoaded,Train_X,condition_names)
        
        #Handle all_trials
        LogISIs = cachedLogISIs[Train_X]
        LogISIs = np.concatenate(LogISIs)
        f,inv_f = training.LogISIsToLikelihoods(LogISIs,bw)
        self.model.all.Likelihood = f
        self.model.all.Inv_Likelihood = inv_f

        #Handle synthetic spiketrain generation
        if synthetic:
            cachedLogISIs = [ [] ]*sessionfile.meta.length_in_trials
            for trial in Train_X:
                cachedLogISIs[trial] = self.synthetic_spiketrain()
            cachedLogISIs = np.array(cachedLogISIs,dtype='object')

        #Handle individual conditions
        LogISIs_per_trial = [len(l) for l in cachedLogISIs[Train_X]]
        total_empty_ISI_trials = np.sum(np.equal(  LogISIs_per_trial,0  ))
        for cond in decoding_conditions:        
            
            LogISIs = cachedLogISIs[decoding_conditions[cond].trials]
            LogISIs = np.concatenate(LogISIs)

            #LogISIs,_ = getLogISIs(sessionfile,clust,decoding_conditions[cond].trials)

            f,inv_f = training.LogISIsToLikelihoods(LogISIs,bw) #This is a gaussian KDE
            self.model.conds[cond].Likelihood = f
            self.model.conds[cond].Inv_Likelihood = inv_f
            self.model.conds[cond].Prior_0 = 1.0 / len(condition_names)

            #Calculate likelihood for 0-ISI trials
            LogISIs_per_trial = [len(l) for l in cachedLogISIs[decoding_conditions[cond].trials]]
            numerator = np.sum(np.equal(LogISIs_per_trial,0)) + 1
            denominator = total_empty_ISI_trials + len(condition_names)
            self.conds[cond].Prior_empty = numerator / denominator
            
        return self.model
    
    def cachedpredictTrial(self, sessionfile,clust,model,trialISIs,conditions = ['target_tone','nontarget_tone'],synthetic = False):
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

    def cachedcalculateAccuracyOnFold(self, sessionfile,clust,model,cachedLogISIs,Test_X,weights,conditions=['target','nontarget'],synthetic=False):
        #Note: this call to getAllConditions uses NO_TRIM all the time because it is only used to determine correctness
        all_conditions = util.getAllConditions(sessionfile,clust,trialsPerDayLoaded='NO_TRIM')
        
        accumulated_correct = 0
        num_correct = 0
        num_total = 0
        num_empty = 0
        
        for trial in Test_X:
            cond,prob,_,empty_ISIs = self.cachedpredictTrial(sessionfile,clust,model,cachedLogISIs[trial],conditions=conditions,synthetic=synthetic)
            
            if cond is None:
                continue

            if np.isin(trial,all_conditions[cond].trials):
                accumulated_correct += weights[cond]
                num_correct += 1
            num_total += 1

            if empty_ISIs:
                num_empty += 1
            
        if num_total <= 0:
            return np.nan,np.nan,np.nan
        else:
            return (num_correct/num_total),(accumulated_correct/num_total),(num_empty/num_total)
        
    def calculateDecodingForSingleNeuron(self,clust,trialsPerDayLoaded,output_directory,trainInterval,testInterval,reps = 1,categories='stimulus'): 

        filename = util.generateDateString(self.loader.meta) + ' cluster ' + str(clust) + ' decoding cached result.pickle'
        filename = os.path.join(output_directory,filename)
        
        try:    
            trainInterval._CalculateAvgLickDelay(self.loader.trials)
            testInterval._CalculateAvgLickDelay(self.loader.trials)
            res = results.cachedCalculateClusterAccuracy(self.loader,clust,trialsPerDayLoaded,trainInterval,testInterval,reps=reps,categories=categories)
        except Exception as e:
            res = results.generateNullResults()
            print(f"session cluster {clust} has thrown error {e}")
            raise e
        try:
            with open(filename, 'wb') as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Problem saving {f} to {filename}. Error: {e}")
                
        print(f"finished with {util.generateDateString(self.loader.meta)} cluster {clust}")
        return res


    











