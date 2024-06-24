import numpy as np 
import pandas as pd 
from KDEpy import FFTKDE 
from scipy.interpolate import interp1d 
from sklearn.model_selection import StratifiedKFold
from decoder.core.model import Model
import pickle
        
class NDecoder:
    def __init__(self, bandwidth, min_ISIs, conditions):
        self.bw = bandwidth
        self.min_ISIs = min_ISIs
        self.conditions = conditions #list of possible labels
        self.model = Model(conditions) 

    def _create_condition_subset_mapping(self, X, y):
        all_conditions = self.conditions
        condition_subset_mapping = {}

        y = np.asarray(y)

        # Check for invalid labels in y
        invalid_labels = set(y) - set(all_conditions)
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {', '.join(invalid_labels)}")

        for condition in all_conditions:
            condition_indices = []
            if len(y) > 0:
                condition_indices = np.where(y == condition)[0]
            condition_subset = X[condition_indices]
            condition_subset_mapping[condition] = condition_subset

        return condition_subset_mapping

    @staticmethod
    def estimate_isi_distribution(log_ISIs, bandwidth):
        """
        Estimate the Inter-Spike Interval (ISI) Distribution using Kernel Density Estimation (KDE).

        Args:
            log_ISIs (numpy.ndarray): An array of log-transformed Inter-Spike Interval (ISI) values.
            bandwidth (float): The bandwidth parameter for the KDE.

        Returns:
            tuple: A tuple containing two functions:
                - A function that interpolates the estimated probability density function (PDF).
                - A function to generate an inverse Cumulative Distribution Function (CDF) for sampling.
        """
        # x = np.linspace(-2, 6, 100)
        # TODO: write x in relation to fftkde data's min and max
        x = np.linspace(-2, 20, 100)

        fftkde = FFTKDE(bw=bandwidth, kernel='gaussian').fit(log_ISIs, weights=None)

        y = fftkde.evaluate(x)

        # Use scipy to interpolate and evaluate on an arbitrary grid
        pdf_interpolator = interp1d(x, y, kind='linear', assume_sorted=True)

        # Generate an inverse CDF for sampling
        norm_y = np.cumsum(y) / np.sum(y)
        norm_y[0] = 0 # return 0 as value if inverse cdf gets 0 as input
        norm_y[-1] = 1 # return biggest value if inverse cdf gets 1 as input
        cdf_inverse_interpolator = interp1d(norm_y, x, kind='linear', assume_sorted=True)

        print("ISI distribution estimated")
        
        return pdf_interpolator, cdf_inverse_interpolator 

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

        print("Created condition subset mapping")

        if len(X) < self.min_ISIs:
            print("Insufficient LogISIs data for analysis.")
            return None
        
        log_ISIs_concat = np.concatenate(X) #flattens it to one array of ISIs

        if len(log_ISIs_concat) < self.min_ISIs:
            print("Insufficient LogISIs data after flattening for analysis.")
            return None

        f,inv_f = self.estimate_isi_distribution(log_ISIs_concat, self.bw)
        self.model.set_all(f, inv_f) 

        # Handle individual conditions
        num_ISIs_per_trial = [len(l) for l in X]
        total_empty_ISI_trials = np.sum(np.equal(num_ISIs_per_trial,0))

        for label in decoding_conditions: 
            logISIs = decoding_conditions[label]
            logISIs_concat = np.concatenate(logISIs)

            if len(logISIs_concat)< self.min_ISIs:
                print(f"Skipping fold. Not enough ISIs for the {label} condition")
                return None
            f,inv_f = self.estimate_isi_distribution(logISIs_concat, self.bw) # This is a gaussian KDE
            prior_0 = 1.0 / len(self.conditions)

            # Calculate likelihood for 0-ISI trials
            num_ISIs_per_trial = [len(l) for l in decoding_conditions[label]]
            numerator = np.sum(np.equal(num_ISIs_per_trial, 0)) + 1 
            denominator = total_empty_ISI_trials + len(self.conditions)
            prior_empty = numerator / denominator 
            self.model.set_cond(label, f, inv_f, prior_0, prior_empty)

        return self.model 

    def _predict_single_trial(self, model, trial_logISIs):
        """
        Predicts the condition for a single trial using calculated probabilities.

        Args:
            model: The model containing condition parameters and priors.
            trial_logISIs: List of inter-stimulus log intervals for the trial.

        Returns:
            predicted condition, probability of predicted condition, probabilities of all conditions, flag indicating if ISIs are empty
        """
        trial_logISIs = np.asarray(trial_logISIs)
        # No ISIs in trial, make a guess based on priors
        if len(trial_logISIs) < 1:
            max_cond = max(self.conditions, key=lambda cond: model.conds[cond].prior_empty)
            all_priors = {cond: model.conds[cond].prior_empty for cond in self.conditions}
            return max_cond, model.conds[max_cond].prior_empty, all_priors, True
        
        # ISIs were present in the trial, predict based on normalized probabilities
        else:
            # Set up probabilities with initial values equal to priors
            probabilities = {cond: np.full(len(trial_logISIs) + 1, np.nan)
                            for cond in self.conditions}
        
            for cond in self.conditions:
                # Calculate cumulative log-likelihood ratio (LLR) after each LogISI
                probabilities[cond] = np.cumsum(
                    np.log10(np.concatenate(([model.conds[cond].prior_0],
                                            model.conds[cond].pdf(trial_logISIs)))))

                # Exponentiate back from log to enable normalization
                probabilities[cond] = np.power(10, probabilities[cond])
        
            # Calculate total probability sum to normalize
            sum_of_probs = np.zeros(len(trial_logISIs) + 1)
            for cond in self.conditions:
                sum_of_probs += probabilities[cond]

            # Normalize all probabilities
            for cond in self.conditions:
                # probabilities[cond] /= sum_of_probs
                probabilities[cond] = np.divide(probabilities[cond], sum_of_probs, 
                                                out= np.zeros_like(probabilities[cond]), 
                                                where= sum_of_probs != 0)


            max_cond = max(self.conditions, key=lambda cond: probabilities[cond][-1])
            all_probs = {cond : probabilities[cond][-1] for cond in probabilities}
            return max_cond, probabilities[max_cond][-1], all_probs, False


    def predict_conditions(self, X):
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
        for trial_logISIs in X:
            pred_condition, _, _, _ = self._predict_single_trial(self.model, trial_logISIs)
            predicted_conditions.append(pred_condition)
        return np.array(predicted_conditions)

    def predict_condition_probs(self, X):
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
        for trial_logISIs in X:
            _, _, trial_probs, _ = self._predict_single_trial(self.model, trial_logISIs)
            probabilities.append([trial_probs[cond] for cond in self.conditions])
        return pd.DataFrame(probabilities, columns=self.conditions)

    # TODO: ask if this is necessary to implement
    def enforce_equal_trials(self, X, y):
        # Enforce an even number of trials of each label if necessary 

        # trials = np.array(trials)
        
        # X = np.ones(len(trials))
        # y = np.ones(len(trials))
        # for idx,trial in enumerate(trials):
        #     if sessionfile.trials.target[trial] and sessionfile.trials.go[trial]:
        #         y[idx] = 1
        #     elif sessionfile.trials.target[trial] and not sessionfile.trials.go[trial]:
        #         y[idx] = 2
        #     elif not sessionfile.trials.target[trial] and sessionfile.trials.go[trial]:
        #         y[idx] = 3
        #     elif not sessionfile.trials.target[trial] and not sessionfile.trials.go[trial]:
        #         y[idx] = 4

        # #Enforce an even number of go and nogo trials
        # y_go_mask   = np.logical_or(np.equal(y,1) , np.equal(y,3))
        # y_nogo_mask = np.logical_or(np.equal(y,2) , np.equal(y,4))
        # ymin = min(np.sum(y_go_mask),np.sum(y_nogo_mask))
        # y_go_idx = np.where(y_go_mask)[0]
        # y_nogo_idx = np.where(y_nogo_mask)[0]
        # idx_go_new = np.random.choice(y_go_idx,ymin,replace=False)
        # idx_nogo_new = np.random.choice(y_nogo_idx,ymin,replace=False)              #Needs to stay in order
        # idx_new = np.concatenate((idx_go_new,idx_nogo_new))
        # X = X[idx_new]
        # y = y[idx_new]

        pass

    def generate_stratified_K_folds(self, X, y, K):
        # X, y = self.encode_equal_trials(X, y)

        train_validate_pairs = []

        X = np.asarray(X)
        y = np.asarray(y)

        skf = StratifiedKFold(n_splits=K, shuffle=True)
        for train_idx, validate_idx in skf.split(X, y):
            train_X = X[train_idx]
            train_y = y[train_idx]
            validate_X = X[validate_idx]
            validate_y = y[validate_idx]

            train_validate_pairs.append(((train_X, train_y), (validate_X, validate_y)))
        return train_validate_pairs

    def calculateAccuracy(self, test_X, test_y):
        # weighted_correct = 0
        num_correct = 0
        num_empty_ISIs = 0
        num_preds = 0
        
        for idx, trial in enumerate(test_X):
            cond, prob_cond, _, empty_ISIs = self._predict_single_trial(self.model, trial)
            
            if cond is None:
                continue

            if cond == test_y[idx]:
                # weighted_correct += weights[cond]
                num_correct += 1
            
            if empty_ISIs:
                num_empty_ISIs += 1

            num_preds += 1
            
        if num_preds <= 0:
            # return np.nan,np.nan,np.nan
            return np.nan,np.nan
        else:
            # return (num_correct/num_preds),(weighted_correct/num_preds),(num_emptyISIs/num_preds)
            return (num_correct/num_preds), (num_empty_ISIs/num_preds)
    
    def save_as_pickle(self, filename):
        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()


    def fit_transform(self, X):
        # Implement your fit_transform logic here
        # Example: transformed_data = self.model.transform(X)
        pass
