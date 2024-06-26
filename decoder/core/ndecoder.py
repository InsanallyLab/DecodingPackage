import numpy as np 
import pandas as pd 
from KDEpy import FFTKDE 
from scipy.interpolate import interp1d 
from sklearn.model_selection import StratifiedKFold
from decoder.core.model import Model
import pickle
from numpy.typing import ArrayLike, NDArray
        
class NDecoder:
    """
    Stores the generated decoder, which uses estimated ISI distributions to
    decode the condition (e.g. stimulus category or behavioral choice) of a 
    trial via Bayesian inference.

    Attributes
    ----------
    self.bw : float
        Stores optimal kernel bandwidth for Kernel Density Estimation methods 
        used to estimate ISI distributions. 
    self.min_ISIs : int
        Stores the minimum number of ISIs that should be present in a trial in 
        order to proceed with estimating the ISI distribution.
    self.conditions : array-like
        Stores all the possible trial conditions. 
        For example, ["target", "non-target"] for stimulus category, or
        ["go", "no-go"] for behavioral choice.
    self.model : Model
        Stores all relevant unconditional and conditional probability 
        information for observing given ISIs.
    """
    def __init__(
        self, 
        bandwidth: float, 
        min_ISIs: int, 
        conditions: ArrayLike):

        self.bw = bandwidth
        self.min_ISIs = min_ISIs
        self.conditions = conditions
        self.model = Model(conditions) 

    def _create_condition_subset_mapping(self, X: ArrayLike, y: ArrayLike) -> dict:
        """
        Maps each trial condition to all log ISIs from trials with that 
        condition.

        Parameters
        ----------
        X : array-like, shape (num trials, num ISIs per trial)
            Stores log ISIs per trial.
        y : array-like, shape (num_trials,)
            Stores the condition for each trial.

        Returns
        -------
        condition_subset_mapping : dict
            A dict mapping each trial condition (str) to all log ISIs from trials
            with that condition (array-like).
        """

        all_conditions = self.conditions
        condition_subset_mapping = {}

        y = np.asarray(y)

        # Check for invalid conditions in y
        invalid_conditions = set(y) - set(all_conditions)
        if invalid_conditions:
            raise ValueError(f"Invalid labels found: {', '.join(invalid_conditions)}")

        for condition in all_conditions:
            condition_indices = []
            if len(y) > 0:
                condition_indices = np.where(y == condition)[0]
            condition_subset = X[condition_indices]
            condition_subset_mapping[condition] = condition_subset

        return condition_subset_mapping

    @staticmethod
    def estimate_isi_distribution(log_ISIs: NDArray, bandwidth: float):
        """
        Estimates Inter-Spike Interval (ISI) distributions using Kernel Density
        Estimation (KDE).

        Parameters
        ----------
        log_ISIs : numpy.ndarray 
            An array of log-transformed ISI values.
        bandwidth : float
            The kernel bandwidth for KDE.

        Returns
        -------
        pdf_interpolator : scipy.interp1d object
            Represents the estimated probability density function (PDF).
        cdf_inverse_interpolator : scipy.interp1d object
            Represents the estimated inverse Cumulative Distribution Function 
            (CDF) for sampling.
        """
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

    def fit(self, X: ArrayLike, y: ArrayLike) -> Model:
        """
        Fits the model to the given ISI data.

        Parameters
        ----------
        X : array-like, shape (num_trials, len_of_isi)
            Log ISIs for each trial.
        y : array-like, shape (num_trials,)
            Condition for each trial.

        Returns
        -------
        model : Model
            The fitted Model object containing likelihood estimates.
        """ 

        decoding_conditions = self._create_condition_subset_mapping(X, y)

        print("Created condition subset mapping")

        if len(X) < self.min_ISIs:
            print("Insufficient LogISIs data for analysis.")
            return None
        
        log_ISIs_concat = np.concatenate(X) # Flattens it to one array of ISIs

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
            # Uses Gaussian KDE
            f,inv_f = self.estimate_isi_distribution(logISIs_concat, self.bw)
            prior_0 = 1.0 / len(self.conditions)

            # Calculate likelihood for 0-ISI trials
            num_ISIs_per_trial = [len(l) for l in decoding_conditions[label]]
            numerator = np.sum(np.equal(num_ISIs_per_trial, 0)) + 1 
            denominator = total_empty_ISI_trials + len(self.conditions)
            prior_empty = numerator / denominator 
            self.model.set_cond(label, f, inv_f, prior_0, prior_empty)

        return self.model 

    def _predict_single_trial(self, trial_logISIs: ArrayLike) -> tuple[str, float, dict, bool]:
        """
        Predicts the condition for a single trial from its log ISIs.

        Parameters
        ----------
        trial_logISIs: array-like 
            All log ISIs for the trial.

        Returns
        -------
        max_cond : str 
            The predicted condition for the trial.
        max_cond_prob : float
            The probability of the predicted condition.
        all_probs : dict 
            Maps each trial condition (str) to the predicted probability of
            the trial having that condition.
        empty_ISIs : bool
            Flag indicating if trial has no ISIs.
        """
        trial_logISIs = np.asarray(trial_logISIs)
        # No ISIs in trial, make a guess based on priors
        if len(trial_logISIs) < 1:
            max_cond = max(self.conditions, key=lambda cond: self.model.conds[cond].prior_empty)
            all_priors = {cond: self.model.conds[cond].prior_empty for cond in self.conditions}
            return max_cond, self.model.conds[max_cond].prior_empty, all_priors, True
        
        # ISIs were present in the trial, predict based on normalized probabilities
        else:
            # Set up probabilities with initial values equal to priors
            probabilities = {cond: np.full(len(trial_logISIs) + 1, np.nan)
                            for cond in self.conditions}
        
            for cond in self.conditions:
                # Calculate cumulative log-likelihood ratio (LLR) after each LogISI
                probabilities[cond] = np.cumsum(
                    np.log10(np.concatenate(([self.model.conds[cond].prior_0],
                                            self.model.conds[cond].pdf(trial_logISIs)))))

                # Exponentiate back from log to enable normalization
                probabilities[cond] = np.power(10, probabilities[cond])
        
            # Calculate total probability sum to normalize
            sum_of_probs = np.zeros(len(trial_logISIs) + 1)
            for cond in self.conditions:
                sum_of_probs += probabilities[cond]

            # Normalize all probabilities
            for cond in self.conditions:
                probabilities[cond] = np.divide(probabilities[cond], 
                                                sum_of_probs, 
                                                out= np.zeros_like(probabilities[cond]), 
                                                where= sum_of_probs != 0)

            max_cond = max(self.conditions, key=lambda cond: probabilities[cond][-1])
            all_probs = {cond : probabilities[cond][-1] for cond in probabilities}
            return max_cond, probabilities[max_cond][-1], all_probs, False


    def predict_conditions(self, X: ArrayLike) -> NDArray:
        """
        Predicts the condition for each trial in X from its log ISIs.

        Parameters
        ----------
        X : array-like, shape (num trials, num ISIs per trial)
            Log ISIs for each trial.

        Returns
        -------
        predicted_conditions : numpy.ndarray, shape (num_trials,)
            Predicted condition for each trial.

        """
        predicted_conditions = []
        for trial_logISIs in X:
            pred_condition, _, _, _ = self._predict_single_trial(trial_logISIs)
            predicted_conditions.append(pred_condition)
        return np.array(predicted_conditions)

    def predict_condition_probs(self, X : ArrayLike) -> pd.DataFrame:
        """
        Predicts the probabilities of each condition for each trial in X from 
        its log ISIs.

        Parameters
        ----------
        X : array-like, shape (num trials, num ISIs per trial)
            Log ISIs for each trial.

        Returns
        -------
        probabilities : DataFrame, shape (num trials, num conditions)
            Predicted probabilities for each condition for each trial.

        """
        probabilities = []
        for trial_logISIs in X:
            _, _, trial_probs, _ = self._predict_single_trial(trial_logISIs)
            probabilities.append([trial_probs[cond] for cond in self.conditions])
        return pd.DataFrame(probabilities, columns=self.conditions)

    @staticmethod
    def generate_stratified_K_folds(X: ArrayLike, y: ArrayLike, K: int) -> list:
        """
        Generates K stratified folds for cross-validation. Returns the data 
        split into K train/validate pairs.

        Parameters
        ----------
        X : array-like, shape (num trials, num ISIs per trial)
            Log ISIs for each trial.
        y : array-like, shape (num trials,)
            Condition of each trial.
        K : int 
            Number of stratified folds to generate.

        Returns
        -------
        train_validate_pairs : list, shape (K, 2)
            Stores K train/validate pairs. Each entry in this list is a tuple 
            with the following format:
            ((training log ISIs, training trial conditions), 
            (validation log ISIs, validation trial conditions)) 
        """
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

    def calculate_accuracy(
        self, 
        test_X: ArrayLike, 
        test_y: ArrayLike) -> tuple[float, float]:
        """
        Calculates the accuracy of the decoder on unseen test trial data.

        Parameters
        ----------
        test_X : array-like, shape (num trials, num ISIs per trial)
            Log ISIs for each trial.
        test_y : array-like, shape (num trials,)
            Condition of each trial.

        Returns
        -------
        accuracy : float
            Number of correct trial predictions divided by the total number of 
            predictions made. 
        frac_empty_ISIs: float
            Number of predictions made for trials with no ISIs divided by the 
            total number of predictions made.
        """
        num_correct = 0
        num_empty_ISIs = 0
        num_preds = 0
        
        for idx, trial in enumerate(test_X):
            cond, _, _, empty_ISIs = self._predict_single_trial(trial)
            
            if cond is None:
                continue

            if cond == test_y[idx]:
                num_correct += 1
            
            if empty_ISIs:
                num_empty_ISIs += 1

            num_preds += 1
            
        if num_preds == 0:
            return np.nan,np.nan
        else:
            return (num_correct/num_preds), (num_empty_ISIs/num_preds)
    
    def save_as_pickle(self, file_path: str):
        """
        Saves the decoder as a pickle file. 

        Parameters
        ----------
        file_path : str
            Path to store generated pickle file at.
        """
        file = open(file_path, 'wb')
        pickle.dump(self, file)
        file.close()