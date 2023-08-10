from core.isi import * 
from types import SimpleNamespace
class NDecoder:
    def __init__(self, bandwidth, min_isis, conditions):
        self.bw = bandwidth
        self.min_isis = min_isis
        self.conditions = conditions #list of conditions for the task variable to decode 

    def _create_condition_subset_mapping(self, X, y):
        all_conditions = self.conditions 
        condition_subset_mapping = {}

        for condition in all_conditions:
            condition_indices = np.where(y == condition)[0]
            condition_subset = X[condition_indices]
            condition_subset_mapping[condition] = condition_subset

        return condition_subset_mapping

    def fit(self, X, y):
        """ 
        params: 
            X: logisis array (num_trials * len_of_isi )
            y: labels or conditions for each trial (num_trials * 1)

        """ 
        cachedLogISIs = self.isi.get_log_isis(train_interval)

        model = SimpleNamespace()
        model.conds = dict()
        model.all = SimpleNamespace()
        for cond in self.conditions:
            model.conds[cond] = SimpleNamespace()
        
        #Determine trials to use for each condition. Uses the conditions structures from
        #ilep to increase modularity and to ensure that all code is always using the same
        #conditions
        decoding_conditions = self.trial.split_by_conditions(train_x, self.conditions)
        
        LogISIs = cachedLogISIs[train_x]
        if len(LogISIs)<5:
            return None
        LogISIs = np.concatenate(LogISIs)
        if len(LogISIs)<5:
            return None

        f,inv_f = ISI.estimate_isi_distribution(LogISIs, bw)
        model.all.Likelihood = f
        model.all.Inv_Likelihood = inv_f

        #Handle individual conditions
        LogISIs_per_trial = [len(l) for l in LogISIs]
        total_empty_ISI_trials = np.sum(np.equal( LogISIs_per_trial,0  ))
        for cond in decoding_conditions:        
            LogISIs = cachedLogISIs[decoding_conditions[cond].trials]
            LogISIs = np.concatenate(LogISIs)
            if len(LogISIs)<5:
                print(f"Skipping fold. Not enough ISIs for the {cond} condition")
                return None
            f,inv_f = ISI.estimate_isi_distribution(LogISIs, bw) #This is a gaussian KDE
            model.conds[cond].Likelihood = f
            model.conds[cond].Inv_Likelihood = inv_f
            model.conds[cond].Prior_0 = 1.0 / len(self.conditions)

            #Calculate likelihood for 0-ISI trials
            LogISIs_per_trial = [len(l) for l in cachedLogISIs[decoding_conditions[cond].trials]]
            numerator = np.sum(np.equal(LogISIs_per_trial,0)) + 1
            denominator = total_empty_ISI_trials + len(self.conditions)
            model.conds[cond].Prior_empty = numerator / denominator
            
        return model

    def fit_transform(self, X):
        # Implement your fit_transform logic here
        # Example: transformed_data = self.model.transform(X)
        pass

    def predict(self, X):
        # Implement your prediction logic here
        # Example: predictions = self.model.predict(X)
        pass

    def predict_proba(self, X):
        # Implement your _predict_proba logic here
        # Example: probabilities = self.model.predict_proba(X)
        pass
