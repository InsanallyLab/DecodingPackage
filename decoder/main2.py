from core.isi import * 
from types import SimpleNamespace

import joblib

class Model:
    def __init__(self, conditions) -> None:
        self.all = Likelihood()
        self.conds = {}
        for label in conditions:
            self.conds[label] = Likelihood()

    def set_all(self, pdf, inv_cdf): 
        self.all.set_val(pdf, inv_cdf) 

    def set_cond(self, cond, pdf, inv_cdf, prior_0, prior_empty): 
        self.conds[cond].set_val(pdf, inv_cdf, prior_0, prior_empty)
 
    def save_model(self, filename):
        joblib.dump(self, filename)

    def __str__(self):
        conds_str = "\n".join(f"{label}: {likelihood}" for label, likelihood in self.conds.items())
        return f"Model:\nAll: {self.all}\nConditions:\n{conds_str}"

class Likelihood:
    def __init__(self) -> None:
        self.pdf = None
        self.inv_cdf = None
        self.prior_0 = None
        self.prior_empty = None

    def set_val(self, pdf=None, inv_cdf=None, prior_0=None, prior_empty=None):
        if pdf is not None:
            self.pdf = pdf
        if inv_cdf is not None:
            self.inv_cdf = inv_cdf
        if prior_0 is not None:
            self.prior_0 = prior_0
        if prior_empty is not None:
            self.prior_empty = prior_empty

    def __str__(self):
        return f"PDF: {self.pdf}, Inv_CDF: {self.inv_cdf}, Prior_0: {self.prior_0}, Prior_Empty: {self.prior_empty}"


        
class NDecoder:
    def __init__(self, bandwidth, min_isis, conditions):
        self.bw = bandwidth
        self.min_isis = min_isis
        self.conditions = conditions #list of possible labels
        self.model = Model(conditions) 

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
        Fit the model to the given data.

        Parameters:
            X : array-like, shape (num_trials, len_of_isi)
                LogISIs array for each trial.
            y : array-like, shape (num_trials,)
                Labels or conditions for each trial.

        Returns:
            model : Model object
                The fitted Model object containing likelihood estimates.

        Raises:
            ValueError: If the input data X or y is not valid.

        Notes:
            This method fits the model to the provided data by estimating ISI distributions,
            priors, and likelihoods for different conditions.
        """ 
        
        decoding_conditions = self._create_condition_subset_mapping(X, y)

        if len(X)< self.min_isis:
            print("Insufficient LogISIs data for analysis.")
            return None

        
        LogISIs = np.concatenate(LogISIs) #flattens it to one array of ISIs
        
        if len(X)< self.min_isis:
            print("Insufficient LogISIs data after flattening for analysis.")
            return None

        f,inv_f = ISI.estimate_isi_distribution(X, self.bw)
        self.model.set_all(f, inv_f) 

        #Handle individual conditions
        LogISIs_per_trial = [len(l) for l in X]
        total_empty_ISI_trials = np.sum(np.equal( LogISIs_per_trial,0))

        for label in decoding_conditions:        
            LogISIs = decoding_conditions[label]
            LogISIs = np.concatenate(LogISIs)
            if len(LogISIs)< self.min_isis:
                print(f"Skipping fold. Not enough ISIs for the {label} condition")
                return None
            f,inv_f = ISI.estimate_isi_distribution(LogISIs, self.bw) #This is a gaussian KDE
            prior_0 = 1.0 / len(self.conditions)

            #Calculate likelihood for 0-ISI trials
            LogISIs_per_trial = [len(l) for l in decoding_conditions[label]]
            numerator = np.sum(np.equal(LogISIs_per_trial,0)) + 1
            denominator = total_empty_ISI_trials + len(self.conditions)
            prior_empty = numerator / denominator 
            self.model.set_cond(label, f, inv_f, prior_0, prior_empty)

        return self.model 

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
