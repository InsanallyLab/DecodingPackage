import random
from typing import Optional, Union
import numpy as np 
import pandas as pd 
from KDEpy import FFTKDE 
from scipy.interpolate import interp1d 
from sklearn.model_selection import StratifiedKFold
from decoder.core.model import Model
import pickle
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                
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
        bw: float, 
        min_ISIs: int, 
        conditions: ArrayLike):

        self.bw = bw
        self.min_ISIs = min_ISIs
        self.conditions = conditions
        self.model = Model(conditions) 
        
        self.log_ISI_construction_set = None
    
    def _create_condition_subset_mapping(
        self, 
        X: ArrayLike, 
        y: ArrayLike) -> dict:
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
    def generate_synthetic_ISIs(
        log_ISIs_construction_set: ArrayLike, 
        trial_start: float, 
        trial_end: float,
        scaling_factor: Union[int, float] = 1000,
        return_final_spikes: bool = False):
        """
        Generates synthetic log ISIs for a given trial. Fills the duration from 
        trial_start to trial_end with randomly sampled log ISIs from the log ISI
        construction set.

        Parameters
        ----------
        log_ISIs_construction_set : array-like, shape (num log ISIs, )
            Real log ISI data. 
        trial_start : float
            Trial start time for the given trial.
        trial_end : float
            Trial end time for the given trial.
        scaling_factor : int, optional
            Factor to scale the spike times. Needs to match the scaling factor
            used in Session.compute_log_ISIs. Defaults to 1000.
        return_final_spikes : bool, optional
            If true, an array of the final (later of the two) spikes 
            corresponding to each synthetic log ISI is also returned.

        Returns
        -------
        synthetic_log_ISIs : array-like, shape (num log ISIs, )
            The synthetic log ISIs based on the given trial data.
        return_final_spikes : array-like, shape (num log ISIs, )
            Only returned if return_final_spikes parameter is True. The second
            of the two spikes that correspond with each synthetic log ISI.
        """
        # Computes trial length, scaled by scaling factor. 
        trial_length = (trial_end - trial_start)*scaling_factor
        
        if len(log_ISIs_construction_set) == 0:
            if return_final_spikes:
                return np.array([]), np.array([])
            return np.array([])

        synthetic_log_ISIs = []
        synthetic_final_spikes = []

        first_log_ISI = random.choice(log_ISIs_construction_set)
        ctime = 10**first_log_ISI
        while True:
            synthetic_log_ISI = random.choice(log_ISIs_construction_set)
            ctime += 10**synthetic_log_ISI
            if ctime <= trial_length:
                synthetic_log_ISIs.append(synthetic_log_ISI)
                synthetic_final_spikes.append(ctime / scaling_factor)
            else:
                break
        
        if return_final_spikes:
            return np.array(synthetic_log_ISIs), np.array(synthetic_final_spikes)
        return np.array(synthetic_log_ISIs)

    @staticmethod
    def estimate_ISI_distribution(log_ISIs: NDArray, bw: float):
        """
        Estimates Inter-Spike Interval (ISI) distributions using Kernel Density
        Estimation (KDE).

        Parameters
        ----------
        log_ISIs : numpy.ndarray 
            An array of log-transformed ISI values.
        bw : float
            The kernel bandwidth for KDE.

        Returns
        -------
        pdf_interpolator : scipy.interp1d object
            Represents the estimated probability density function (PDF).
        cdf_inverse_interpolator : scipy.interp1d object
            Represents the estimated inverse Cumulative Distribution Function 
            (CDF) for sampling.
        """
        x = np.linspace(-2, 18, 100)
        
        fftkde = FFTKDE(bw=bw, kernel='gaussian').fit(log_ISIs, weights=None)

        y = fftkde.evaluate(x)

        # Use scipy to interpolate and evaluate on an arbitrary grid
        pdf_interpolator = interp1d(x, y, kind='linear', assume_sorted=True)

        # Generate an inverse CDF for sampling
        norm_y = np.cumsum(y) / np.sum(y)
        norm_y[0] = 0 # return 0 as value if inverse cdf gets 0 as input
        norm_y[-1] = 1 # return biggest value if inverse cdf gets 1 as input
        cdf_inverse_interpolator = interp1d(norm_y, x, kind='linear', assume_sorted=True)
        
        return pdf_interpolator, cdf_inverse_interpolator 

    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        synthetic: bool = False,
        start_end_times : Optional[ArrayLike] = None):
        """
        Fits the model to the given ISI data.

        Parameters
        ----------
        X : array-like, shape (num trials, num ISIs per trial)
            Log ISIs for each trial.
        y : array-like, shape (num trials,)
            Condition for each trial.
        synthetic : bool, default False
            Whether synthetic log ISI data should be constructed and used to
            fit the model.
        start_end_times : array-like, shape (num trials, 2), optional
            The start and end times for each trial, to be used when
            generating synthetic log ISIs. This arg is required if 
            synthetic = True. 
        """ 
        if len(X) < self.min_ISIs:
            print("Insufficient LogISIs data for analysis.")
            return
        
        log_ISIs_concat = np.concatenate(X) # Flattens it to one array of ISIs

        if len(log_ISIs_concat) < self.min_ISIs:
            print("Insufficient LogISIs data after flattening for analysis.")
            return 

        f,inv_f = self.estimate_ISI_distribution(log_ISIs_concat, self.bw)
        self.model.set_all(pdf=f, inv_cdf=inv_f) 

        # Handle synthetic log ISI generation
        if synthetic:
            if start_end_times is None:
                raise ValueError("Trial start and end times must be passed in to generate synthetic log ISI data")

            synthetic_log_ISIs = []
            for idx, trial_log_ISIs in enumerate(X):
                trial_start, trial_end = start_end_times[idx]
                trial_synthetic_log_ISIs = self.generate_synthetic_ISIs(
                    log_ISIs_concat, 
                    trial_start,
                    trial_end)
                synthetic_log_ISIs.append(trial_synthetic_log_ISIs)

            X = np.array(synthetic_log_ISIs,dtype='object')

        # Handle individual conditions
        num_ISIs_per_trial = [len(l) for l in X]
        total_empty_ISI_trials = np.sum(np.equal(num_ISIs_per_trial,0))

        decoding_conditions = self._create_condition_subset_mapping(X, y)

        for label in decoding_conditions: 
            prior_0 = 1.0 / len(self.conditions)

            # Calculate likelihood for 0-ISI trials
            num_ISIs_per_trial = [len(l) for l in decoding_conditions[label]]
            numerator = np.sum(np.equal(num_ISIs_per_trial, 0)) + 1 
            denominator = total_empty_ISI_trials + len(self.conditions)
            prior_empty = numerator / denominator 
            
            self.model.set_cond(
                cond=label, 
                prior_0=prior_0,
                prior_empty=prior_empty)

            log_ISIs = decoding_conditions[label]
            log_ISIs_concat = np.concatenate(log_ISIs)

            if len(log_ISIs_concat) < self.min_ISIs:
                # print(f"Skipping fold. Not enough ISIs for the {label} condition")
                continue 

            # Uses Gaussian KDE
            f,inv_f = self.estimate_ISI_distribution(log_ISIs_concat, self.bw)
            self.model.set_cond(
                cond=label, 
                pdf=f, 
                inv_cdf=inv_f)

    def fit_window(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        windows: ArrayLike,
        synthetic: bool = False):
        """
        Fits the model to the given ISI data using sliding windows. 

        Parameters
        ----------
        X : array-like, shape (num trials, num windows per trial, num ISIs per window)
            Log ISIs for each sliding window in each trial.
        y : array-like, shape (num trials,)
            Condition for each trial.
        windows : array-like, shape (num windows, 2)
            The start and end times (time-locked to trial starts) of each
            window to compute PDFs for. 
        synthetic : bool, default False
            Whether synthetic log ISI data should be constructed and used to
            fit the model.
        """ 
        log_ISI_construction_set = []

        window_pdfs = {}
        window_inv_cdfs = {}

        for window_idx, (w_start, w_end) in enumerate(windows):
            window_ISIs_concat = []

            for trial in X:
                if window_idx < len(trial):
                    window_ISIs_concat.extend(trial[window_idx])
                
            window_ISIs_concat = np.array(window_ISIs_concat)

            if len(window_ISIs_concat) > 0:
                log_ISI_construction_set.extend(window_ISIs_concat)

            if len(window_ISIs_concat) < self.min_ISIs:
                # print("Insufficient LogISIs data for analysis.")
                continue

            # f,inv_f = self.estimate_ISI_distribution(window_ISIs_concat, self.bw['overall'])
            f,inv_f = self.estimate_ISI_distribution(window_ISIs_concat, self.bw)

            window_pdfs[(w_start, w_end)] = f
            window_inv_cdfs[(w_start, w_end)] = inv_f

        self.model.set_all(pdf=window_pdfs, inv_cdf=window_inv_cdfs)

        # Handle synthetic log ISI generation
        if synthetic:
            log_ISI_construction_set = np.array(log_ISI_construction_set)
            self.log_ISI_construction_set = log_ISI_construction_set

            synthetic_log_ISIs = []

            for trial in X:
                synthetic_trial_log_ISIs = []

                for window_idx, window_ISIs in enumerate(trial):
                    w_start, w_end = windows[window_idx]

                    window_synthetic_log_ISIs = self.generate_synthetic_ISIs(
                        log_ISI_construction_set, 
                        w_start,
                        w_end)
                    synthetic_trial_log_ISIs.append(window_synthetic_log_ISIs)
                
                synthetic_log_ISIs.append(np.array(synthetic_trial_log_ISIs, dtype='object'))

            X = np.array(synthetic_log_ISIs,dtype='object')

        total_empty_ISI_trials = 0
        for trial in X:
            if np.all([len(window) == 0 for window in trial]):
                total_empty_ISI_trials += 1

        condition_to_X = self._create_condition_subset_mapping(X, y)

        # Handle individual conditions
        for label in condition_to_X: 
            X_cond = condition_to_X[label]

            num_empty_ISIs_cond = 0
            for trial in X_cond:
                if np.all([len(window) == 0 for window in trial]):
                    num_empty_ISIs_cond += 1
            
            prior_0 = 1.0 / len(self.conditions)

            # Calculate likelihood for 0-ISI trials
            numerator = num_empty_ISIs_cond + 1 
            denominator = total_empty_ISI_trials + len(self.conditions)
            prior_empty = numerator / denominator 

            self.model.set_cond(
                cond=label, 
                prior_0=prior_0,
                prior_empty=prior_empty)

            window_pdfs = {}
            window_inv_cdfs = {}

            for window_idx, (w_start, w_end) in enumerate(windows):
                window_ISIs_concat = []

                for trial in X_cond:
                    if window_idx < len(trial):
                        window_ISIs_concat.extend(trial[window_idx])
                
                window_ISIs_concat = np.array(window_ISIs_concat)

                if len(window_ISIs_concat) < self.min_ISIs:
                    # print(f"Skipping fold. Not enough ISIs for the {label} condition")
                    continue

                # f,inv_f = self.estimate_ISI_distribution(window_ISIs_concat, self.bw[label])
                f,inv_f = self.estimate_ISI_distribution(window_ISIs_concat, self.bw)

                window_pdfs[(w_start, w_end)] = f
                window_inv_cdfs[(w_start, w_end)] = inv_f 

            self.model.set_cond(
                cond=label, 
                pdf=window_pdfs, 
                inv_cdf=window_inv_cdfs)

    def _spike_to_window(
        self,
        spike: float,
        windows: ArrayLike, 
        metric: str = "center"):
        """
        Returns the best sliding window for the given spike based on the metric.

        Parameters
        ----------
        spike : float
            The spike to find the best window for. 
        windows : array-like, shape (num_windows, 2)
            Start and end times (time-locked to trial starts) of each window.
        metric : str (default = 'center')
            The metric to select the best sliding window:
            'center': return the sliding window with its center closest to
                the given spike. 
            'start': return the sliding window with its start time closest
                to the given spike. 
            'end': return the sliding window with its end time closest
                to the given spike. 

        Returns
        -------
        window : (float, float) or (None, None)
            The start and end time of the best sliding window for the spike.
        """ 
        best_window = (None, None)

        if metric == 'center':
            max_dist = 0.
            for start, end in windows:
                if spike >= start and spike <= end:
                    curr_dist = min(end - spike, spike - start)
                    if curr_dist > max_dist:
                        max_dist = curr_dist
                        best_window = (start, end)

        elif metric == 'start':
            min_dist = None 
            for start, end in windows:
                if spike >= start and spike <= end:
                    curr_dist = spike - start
                    if min_dist == None or curr_dist < min_dist:
                        min_dist = curr_dist
                        best_window = (start, end)

        elif metric == 'end':
            min_dist = None 
            for start, end in windows:
                if spike >= start and spike <= end:
                    curr_dist = end - spike
            if min_dist == None or curr_dist < min_dist:
                min_dist = curr_dist
                best_window = (start, end)

        return best_window

    def predict_window_single_trial(
        self, 
        trial_log_ISIs: ArrayLike, 
        trial_final_spikes: ArrayLike,
        synthetic : bool = False,
        display_trial: bool = False,
        best_window_metric: str = 'center',
        log_ISIs_construction_set : Optional[ArrayLike] = None,
        trial_start : Optional[float] = None,
        trial_end : Optional[float] = None,
        file_path: Optional[str] = None) -> tuple[str, float, dict, bool]:
        """
        Predicts the condition for a single trial from its log ISIs using sliding
        windows.

        Parameters
        ----------
        trial_logISIs : array-like 
            All log ISIs for the trial.
        trial_final_spikes : float
            The final spike for the log ISIs in each trial. Used to select which 
            sliding window's PDF should be used in predictions.
        synthetic : bool, default False
            Whether synthetic spike train data should be constructed and used in
            model prediction.
        display_trial : bool, default False
            Whether to display a figure showing the prediction of the model 
            on the single trial. 
        best_window_metric : str
            The metric to use to select the best sliding window for the final spike.
        log_ISIs_construction_set : ArrayLike, optional
            The set of real log ISI data to be used when generating synthetic log ISIs.
            This arg is required if synthetic = True.
        trial_start : float, optional
            The trial start time, to be used when generating synthetic log ISIs. 
            This arg is required if synthetic = True. 
        trial_end : float, optional
            The trial end time, to be used when generating synthetic log ISIs. 
            This arg is required if synthetic = True. 
        file_path : str, optional 
            The file path to (optionally) save the trial figure at.

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

        if best_window_metric not in ['center', 'start', 'end']:
            raise ValueError("Invalid window metric. The only valid metrics are 'center', 'start', and 'end'.")

        trial_log_ISIs = np.asarray(trial_log_ISIs)
        trial_final_spikes = np.asarray(trial_final_spikes)

        # Handle synthetic trial generation
        if synthetic:
            trial_log_ISIs, trial_final_spikes = self.generate_synthetic_ISIs(
                log_ISIs_construction_set, 
                trial_start, 
                trial_end, 
                return_final_spikes=True)

        # No ISIs in trial, make a guess based on priors
        if len(trial_log_ISIs) < 1:
            sum_of_probs = 0
            for cond in self.conditions:
                if self.model.conds[cond].prior_empty is None:
                    raise ValueError("Model's prior_empty uninitialized for %s condition." %cond)

                sum_of_probs += self.model.conds[cond].prior_empty

            all_priors = {cond: self.model.conds[cond].prior_empty / sum_of_probs for cond in self.conditions}   
            max_cond = max(self.conditions, key=lambda cond: all_priors[cond])
            return max_cond, all_priors[max_cond], all_priors, True
        
        # ISIs were present in the trial, predict based on normalized probabilities
        else:
            # Error handling for model priors and PDFs
            for cond in self.conditions:
                if self.model.conds[cond].prior_0 is None:
                    raise ValueError("Model's prior_0 uninitialized for %s condition." %cond)
                    
                if self.model.conds[cond].pdf is None:
                    raise ValueError("Model's pdf uninitialized for %s condition." %cond)

            # Map each condition to all the windows with computed PDFs
            windows = {cond: np.sort(list(self.model.conds[cond].pdf.keys())) 
                        for cond in self.conditions}

            # Set up probabilities with initial values equal to priors
            probabilities = {cond: np.full(len(trial_log_ISIs) + 1, np.nan)
                            for cond in self.conditions}
            for cond in self.conditions:
                probabilities[cond][0] = self.model.conds[cond].prior_0

            # Update probabilities for conditions with each log ISI
            for idx, final_spike in enumerate(trial_final_spikes):
                spike_out_of_bounds = False

                for cond in self.conditions:
                    best_window = self._spike_to_window(final_spike, windows[cond], best_window_metric)
                    if best_window == (None, None):
                        spike_out_of_bounds = True
                        break

                    best_pdf = self.model.conds[cond].pdf[best_window]
                    probabilities[cond][idx + 1] = best_pdf([trial_log_ISIs[idx]])

                if spike_out_of_bounds:
                    for cond in self.conditions:
                        probabilities[cond][idx + 1] = probabilities[cond][idx]
            
            sum_of_probs = np.zeros(len(trial_log_ISIs) + 1)
            for cond in self.conditions:
                # Calculate cumulative log-likelihood ratio (LLR) after each log ISI
                probabilities[cond] = np.cumsum(
                    np.log10(probabilities[cond]))

                # Exponentiate back from log to enable normalization
                probabilities[cond] = np.power(10, probabilities[cond])

                # Calculate total probability sum to normalize
                sum_of_probs += probabilities[cond]

            # Normalize all probabilities
            for cond in self.conditions:
                probabilities[cond] = np.divide(probabilities[cond], 
                                                sum_of_probs, 
                                                out= np.zeros_like(probabilities[cond]), 
                                                where= sum_of_probs != 0)

            # Displays probabilities as trial progresses
            if display_trial: 
                x = [i for i in range(len(trial_log_ISIs) + 1)]
                for cond in self.conditions:
                    plt.plot(x, probabilities[cond], label=cond)

                plt.ylabel('P(stimulus)')
                plt.xlabel('Number of ISIs seen')
                plt.legend()
                
                if file_path is not None:
                    plt.savefig(file_path)

                plt.show()

            max_cond = max(self.conditions, key=lambda cond: probabilities[cond][-1])
            all_probs = {cond : probabilities[cond][-1] for cond in probabilities}
            return max_cond, probabilities[max_cond][-1], all_probs, False

    def predict_single_trial(
        self, 
        trial_log_ISIs: ArrayLike, 
        synthetic : bool = False,
        display_trial: bool = False,
        log_ISIs_construction_set : Optional[ArrayLike] = None,
        trial_start : Optional[float] = None,
        trial_end : Optional[float] = None,
        file_path: Optional[str] = None) -> tuple[str, float, dict, bool]:
        """
        Predicts the condition for a single trial from its log ISIs.

        Parameters
        ----------
        trial_logISIs: array-like 
            All log ISIs for the trial.
        synthetic : bool, default False
            Whether synthetic spike train data should be constructed and used in
            model prediction.
        display_trial : bool, default False
            Whether to display a figure showing the prediction of the model 
            on the single trial. 
        log_ISIs_construction_set : ArrayLike, optional
            The set of real log ISI data to be used when generating synthetic log ISIs.
            This arg is required if synthetic = True.
        trial_start : float, optional
            The trial start time, to be used when generating synthetic log ISIs. 
            This arg is required if synthetic = True. 
        trial_end : float, optional
            The trial end time, to be used when generating synthetic log ISIs. 
            This arg is required if synthetic = True. 
        file_path : str, optional 
            The file path to (optionally) save the trial figure at.

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
        trial_log_ISIs = np.asarray(trial_log_ISIs)

        # Handle synthetic log ISI generation
        if synthetic:
            trial_log_ISIs = self.generate_synthetic_ISIs(
                log_ISIs_construction_set, 
                trial_start, 
                trial_end)

        # No ISIs in trial, make a guess based on priors
        if len(trial_log_ISIs) < 1:
            sum_of_probs = 0
            
            for cond in self.conditions:
                if self.model.conds[cond].prior_empty is None:
                    raise ValueError("Model's prior_empty uninitialized for %s condition." %cond)
                sum_of_probs += self.model.conds[cond].prior_empty

            all_priors = {cond: self.model.conds[cond].prior_empty / sum_of_probs for cond in self.conditions}   
            max_cond = max(self.conditions, key=lambda cond: all_priors[cond])
            return max_cond, all_priors[max_cond], all_priors, True
        
        # ISIs were present in the trial, predict based on normalized probabilities
        else:
            # Set up probabilities with initial values equal to priors
            probabilities = {cond: np.full(len(trial_log_ISIs) + 1, np.nan)
                            for cond in self.conditions}
        
            for cond in self.conditions:
                if self.model.conds[cond].prior_0 is None:
                    raise ValueError("Model's prior_0 uninitialized for %s condition." %cond)

                if self.model.conds[cond].pdf is None:
                    # print("Model's PDF uninitialized for %s condition." %cond)
                    zero_pdf = lambda x : [0 for i in range(len(x))]
                    probabilities[cond] = np.cumsum(
                        np.log10(np.concatenate(([self.model.conds[cond].prior_0],
                                                zero_pdf(trial_log_ISIs)))))
                else:
                    # Calculate cumulative log-likelihood ratio (LLR) after each LogISI
                    probabilities[cond] = np.cumsum(
                        np.log10(np.concatenate(([self.model.conds[cond].prior_0],
                                                self.model.conds[cond].pdf(trial_log_ISIs)))))

                # Exponentiate back from log to enable normalization
                probabilities[cond] = np.power(10, probabilities[cond])
        
            # Calculate total probability sum to normalize
            sum_of_probs = np.zeros(len(trial_log_ISIs) + 1)
            for cond in self.conditions:
                sum_of_probs += probabilities[cond]

            # Normalize all probabilities
            for cond in self.conditions:
                probabilities[cond] = np.divide(probabilities[cond], 
                                                sum_of_probs, 
                                                out= np.zeros_like(probabilities[cond]), 
                                                where= sum_of_probs != 0)

            # Displays probabilities as trial progresses
            if display_trial: 
                x = [i for i in range(len(trial_log_ISIs) + 1)]
                for cond in self.conditions:
                    plt.plot(x, probabilities[cond], label=cond)

                plt.ylabel('P(stimulus)')
                plt.xlabel('Number of ISIs seen')
                plt.legend()
                
                if file_path is not None:
                    plt.savefig(file_path)

                plt.show()

            max_cond = max(self.conditions, key=lambda cond: probabilities[cond][-1])
            all_probs = {cond : probabilities[cond][-1] for cond in probabilities}
            return max_cond, probabilities[max_cond][-1], all_probs, False

    @staticmethod
    def generate_stratified_K_folds(
        X: ArrayLike, 
        y: ArrayLike, 
        K: int,
        return_indices: bool = False) -> list:
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
        return_indices : bool (default = False)
            If true, this function also returns the indices for all data kept in
            each train/val split.

        Returns
        -------
        train_validate_pairs : list, shape (K, 2)
            Stores K train/validate pairs. Each entry in this list is a tuple 
            with the following format:
            ((training log ISIs, training trial conditions), 
            (validation log ISIs, validation trial conditions)) 

        fold_idx : returned if return_indices is true
            array-like, shape (K, 2, num indices kept in train/val dataset)
            The indices for all data kept in each train/val split.
        """
        train_validate_pairs = []

        fold_idx = []

        X = np.asarray(X)
        y = np.asarray(y)

        skf = StratifiedKFold(n_splits=K, shuffle=True)
        for train_idx, validate_idx in skf.split(X, y):
            train_X = X[train_idx]
            train_y = y[train_idx]
            validate_X = X[validate_idx]
            validate_y = y[validate_idx]

            train_validate_pairs.append(((train_X, train_y), (validate_X, validate_y)))

            if return_indices:
                fold_idx.append((train_idx, validate_idx))
        
        if return_indices:
            return train_validate_pairs, fold_idx

        return train_validate_pairs

    def calculate_accuracy(
        self, 
        test_X: ArrayLike, 
        test_y: ArrayLike, 
        synthetic: bool = False,
        start_end_times: Optional[ArrayLike] = None) -> tuple[float, float]:
        """
        Calculates the accuracy of the decoder on unseen test trial data.

        Parameters
        ----------
        test_X : array-like, shape (num trials, num ISIs per trial)
            Log ISIs for each trial.
        test_y : array-like, shape (num trials,)
            Condition of each trial.
        synthetic : bool, default False
            Whether synthetic spike train data should be constructed and used in
            model prediction.
        start_end_times : array-like, shape (num_trials, 2)
            The start and end times for each trial, to be used when
            generating synthetic log ISIs. This arg is required if 
            synthetic = True. 

        Returns
        -------
        accuracy : float
            Number of correct trial predictions divided by the total number of 
            predictions made. 
        frac_empty_ISIs: float
            Number of predictions made for trials with no ISIs divided by the 
            total number of predictions made.
        """
        cond_prob_correct = {}
        cond_num_preds = {}

        num_empty_ISIs = 0
        total_num_preds = 0

        if synthetic and start_end_times is None:
            raise ValueError("Trial start and end times must be passed in to generate synthetic log ISI data")
        
        for idx, trial in enumerate(test_X):
            test_X_concat = np.concatenate(test_X)

            trial_start, trial_end, log_ISI_construction_set = None, None, None

            if synthetic:
                trial_start, trial_end = start_end_times[idx]
                log_ISI_construction_set = test_X_concat

            _, _, all_probs, empty_ISIs = self.predict_single_trial(
                trial, 
                synthetic=synthetic,
                log_ISIs_construction_set=log_ISI_construction_set,
                trial_start=trial_start, 
                trial_end=trial_end)
            
            if test_y[idx] not in cond_prob_correct:
                cond_prob_correct[test_y[idx]] = 0
            
            if test_y[idx] not in cond_num_preds:
                cond_num_preds[test_y[idx]] = 0

            cond_prob_correct[test_y[idx]] = cond_prob_correct[test_y[idx]] + all_probs[test_y[idx]]
            cond_num_preds[test_y[idx]] = cond_num_preds[test_y[idx]] + 1
            
            if empty_ISIs:
                num_empty_ISIs += 1

            total_num_preds += 1

        if total_num_preds == 0:
            return np.nan,np.nan

        balanced_acc = 0
        num_conds = len(cond_prob_correct.keys())
        for cond in cond_prob_correct:
            balanced_acc += (cond_prob_correct[cond] / cond_num_preds[cond])
        balanced_acc /= num_conds
        return balanced_acc, (num_empty_ISIs/total_num_preds)
    
    def calculate_window_accuracy(
        self, 
        test_X: ArrayLike, 
        test_y: ArrayLike, 
        test_final_spikes: ArrayLike,
        synthetic: bool = False,
        start_end_times: Optional[ArrayLike] = None) -> tuple[float, float]:
        """
        Calculates the balanced accuracy of the decoder on unseen test trial 
        data using sliding windows.

        Parameters
        ----------
        test_X : array-like, shape (num trials, num ISIs per trial)
            Log ISIs for each trial.
        test_y : array-like, shape (num trials,)
            Condition of each trial.
        trial_final_spikes : float
            The final spike for each log ISI in each trial. Used to select which sliding window's
            PDF should be used in predictions.
        synthetic : bool, default False
            Whether synthetic spike train data should be constructed and used in
            model prediction.
        start_end_times : array-like, shape (num_trials, 2)
            The start and end times for each trial, to be used when
            generating synthetic log ISIs. This arg is required if 
            synthetic = True. 

        Returns
        -------
        accuracy : float
            Number of correct trial predictions divided by the total number of 
            predictions made. 
        frac_empty_ISIs: float
            Number of predictions made for trials with no ISIs divided by the 
            total number of predictions made.
        """
        cond_prob_correct = {}
        cond_num_preds = {}

        num_empty_ISIs = 0
        total_num_preds = 0
            
        if synthetic and start_end_times is None:
            raise ValueError("Trial start and end times must be passed in to generate synthetic log ISI data")

        test_X_concat = np.concatenate(test_X)

        for idx, trial in enumerate(test_X):
            trial_start, trial_end, log_ISI_construction_set = None, None, None

            if synthetic:
                trial_start, trial_end = start_end_times[idx]
                log_ISI_construction_set = test_X_concat

            _, _, all_probs, empty_ISIs = self.predict_window_single_trial(
                trial,
                test_final_spikes[idx],
                synthetic=synthetic, 
                log_ISIs_construction_set=log_ISI_construction_set,
                trial_start=trial_start, 
                trial_end=trial_end)

            if test_y[idx] not in cond_prob_correct:
                cond_prob_correct[test_y[idx]] = 0
            
            if test_y[idx] not in cond_num_preds:
                cond_num_preds[test_y[idx]] = 0

            cond_prob_correct[test_y[idx]] = cond_prob_correct[test_y[idx]] + all_probs[test_y[idx]]
            cond_num_preds[test_y[idx]] = cond_num_preds[test_y[idx]] + 1
            
            if empty_ISIs:
                num_empty_ISIs += 1

            total_num_preds += 1

        if total_num_preds == 0:
            return np.nan,np.nan

        balanced_acc = 0
        num_conds = len(cond_prob_correct.keys())
        for cond in cond_prob_correct:
            balanced_acc += (cond_prob_correct[cond] / cond_num_preds[cond])
        balanced_acc /= num_conds
        return balanced_acc, (num_empty_ISIs/total_num_preds)
    
    @staticmethod
    def get_threshold_mask(acc, pval_s):
        """
        Returns a mask for which neurons are statistically significant based on 
        test accuracies and p values. 

        Parameters
        ----------
        acc : array-like, shape (num neurons, )
            Stores mean test accuracies for each neuron.
        pval_s : array-like, shape (num neurons, )
            Stores p values for each neuron computed with respect to a decoder
            trained on synthetic spike trains. 

        Returns
        -------
        is_SS : array-like, shape (num neurons, )
            A boolean array. is_TE[idx] stores if the neuron at that index is
            statistically significant. 
        acc_threshold : float
            The accuracy threshold used on the neurons. 
        """
        acc = np.array(acc)
        pval_s = np.array(pval_s)
        
        low_decoders = np.less(acc,0.5)
        low_accuracies = acc[low_decoders]
        low_magnitudes = np.absolute(low_accuracies-0.5)

        low_magnitudes = np.sort(low_magnitudes)
        if len(low_magnitudes) == 0:
            acc_threshold = 0.5
        else:
            acc_threshold = 0.5+low_magnitudes[int(0.95 * len(low_magnitudes))]

        is_finite = np.isfinite(acc)
        is_valid_pval = np.less(pval_s,0.05)
        # is_above_acc = np.greater_equal(acc, acc_threshold)
        is_above_acc = np.greater_equal(acc,0.5)
        is_SS = np.logical_and(is_valid_pval, is_finite)
        is_SS = np.logical_and(is_SS, is_above_acc)
        return is_SS, acc_threshold
    
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
    
    def display_condition_pdfs(self, file_path: Optional[str] = None):
        """
        Displays and (optionally) saves a figure of the PDFs estimated for each
        condition.

        Parameters
        ----------
        file_path : str
            Path to store generated figure at.
        """
        xs = np.linspace(1.0, 5.0, 1000)
        ys_per_cond = {}
        for cond in self.conditions:
            ys_per_cond[cond] = [self.model.conds[cond].pdf(x) for x in xs]

        for cond in self.conditions:
            plt.plot(xs, ys_per_cond[cond], label=cond)

        plt.xlabel('Log ISI (ms)')
        plt.ylabel('Probability density')
        plt.legend()

        if file_path is not None:
            plt.savefig(file_path)
        plt.show()
    
    def calculate_WLLR(self, x: float, above_cond: str, below_cond: str):
        """
        Computes the weighted LLR for a given ISI. If the value is above 0, the 
        ISI suggests the above_cond, and if the value is below 0, the ISI 
        suggests the below_cond. 

        Parameters
        ----------
        file_path : str
            Path to store generated figure at.
        """
        model = self.model
        p_ISI = model.all.pdf(x)
        if above_cond not in self.conditions:
            raise ValueError("Invalid condition passed in for above_cond")
        if below_cond not in self.conditions:
            raise ValueError("Invalid condition passed in for below_cond")

        p_ISI_given_above = model.conds[above_cond].pdf(x)
        p_ISI_given_below = model.conds[below_cond].pdf(x)
        return p_ISI*(np.log10(p_ISI_given_above) - np.log10(p_ISI_given_below))

    def display_WLLR(self, above_cond: str, below_cond: str, file_path: Optional[str] = None):
        """
        Displays and (optionally) saves a figure of the weighted log-likelihood
        ratio. If the curve is above 0, the ISI suggests the above_cond, and if
        the curve is below 0, the ISI suggests the below_cond. 

        Parameters
        ----------
        file_path : str
            Path to store generated figure at.
        """
        llr_x = np.linspace(1.0, 5.0, 100)
        llr_y = [self.calculate_WLLR(x=x, above_cond=above_cond, below_cond=below_cond) for x in llr_x]

        plt.plot(llr_x, llr_y, 'b')
        plt.plot(llr_x, [0 for i in llr_x], 'k')
        plt.figtext(0.15, 0.2, below_cond)
        plt.figtext(0.15, 0.8, above_cond)
        plt.xlabel('Log ISI (ms)')
        plt.ylabel('W. LLR')
        plt.legend()

        if file_path is not None:
            plt.savefig(file_path)
        plt.show()
