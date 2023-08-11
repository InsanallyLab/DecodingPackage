import numpy as np 
import pandas as pd 
import joblib
from .core.isi import ISI 


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

        # Check for invalid labels in y
        invalid_labels = set(y) - set(all_conditions)
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {', '.join(invalid_labels)}")

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

    def _predict_single_trial(self, model, trialISIs):
        """
        Predicts the condition for a single trial using calculated probabilities.

        Args:
            model: The model containing condition parameters and priors.
            trialISIs: List of inter-stimulus intervals for the trial.

        Returns:
            predicted condition, probability of predicted condition, flag indicating if ISIs are empty
        """
        LogISIs = trialISIs
        
        # Set up probabilities with initial values equal to priors
        probabilities = {cond: np.full(len(LogISIs) + 1, np.nan)
                        for cond in self.conditions}
        
        for cond in self.conditions:
            # Calculate cumulative log-likelihood ratio (LLR) after each LogISI
            probabilities[cond] = np.cumsum(
                np.log10(np.concatenate(([model.conds[cond].Prior_0],
                                        model.conds[cond].Likelihood(LogISIs))))
            )
            # Exponentiate back from log to enable normalization
            probabilities[cond] = np.power(10, probabilities[cond])
        
        # Calculate total probability sum to normalize
        sum_of_probs = np.zeros(len(LogISIs) + 1)
        for cond in self.conditions:
            sum_of_probs += probabilities[cond]
        
        # Normalize all probabilities
        for cond in self.conditions:
            probabilities[cond] /= sum_of_probs
        
        # No ISIs in trial, make a guess based on priors
        if len(LogISIs) < 1:
            maxCond = max(self.conditions, key=lambda cond: model.conds[cond].Prior_empty)
            return maxCond, model.conds[maxCond].Prior_empty, True
        # ISIs were present in the trial, predict based on normalized probabilities
        else:
            maxCond = max(self.conditions, key=lambda cond: probabilities[cond][-1])
            return maxCond, probabilities[maxCond][-1], False


    def predict(self, X):
        """
        Predict the condition for each trial in X.

        Parameters:
            X : array-like, shape (num_trials, len_of_isi)
                LogISIs array for multiple trials.

        Returns:
            predicted_conditions : numpy array, shape (num_trials,)
                Predicted conditions for each trial.

        """
        predicted_conditions = []
        for trialISIs in X:
            pred_condition, _, _, _ = self._predict_single_trial(self.model, trialISIs)
            predicted_conditions.append(pred_condition)
        return np.array(predicted_conditions)

    def predict_proba(self, X):
        """
        Predict the probabilities of each condition for each trial in X.

        Parameters:
            X : array-like, shape (num_trials, len_of_isi)
                LogISIs array for multiple trials.

        Returns:
            probabilities : DataFrame, shape (num_trials, num_conditions)
                Predicted probabilities for each condition for each trial.

        """
        probabilities = []
        for trialISIs in X:
            _, _, trial_probs, _ = self._predict_single_trial(self.model, trialISIs)
            probabilities.append([trial_probs[cond] for cond in self.conditions])
        return pd.DataFrame(probabilities, columns=self.conditions)
