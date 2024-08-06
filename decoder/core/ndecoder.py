import random
from typing import Optional
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
        trial_log_ISIs: ArrayLike, 
        trial_start: float, 
        trial_end: float,
        lead_time: float = 1000):
        """
        Generates synthetic log ISIs for a given trial. Fills the duration from 
        trial_start to trial_end with randomly picked log ISIs.

        Parameters
        ----------
        trial_log_ISIs : array-like, shape (num log ISIs, )
            Real log ISI data for a given trial.
        trial_start : float
            Trial start time for the given trial, in seconds.
        trial_end : float
            Trial end time for the given trial, in seconds.
        lead_time : float (default = 1000)
            Lead time in milliseconds. This lead time is added to trial_start 
            when filling the duration from trial_start to trial_end.

        Returns
        -------
        synthetic_log_ISIs : array-like, shape (num log ISIs, )
            The synthetic log ISIs based on the given trial data.
        """
        # Computes trial length in milliseconds. 
        trial_length = (trial_end - trial_start)*1000
        
        if len(trial_log_ISIs) == 0:
            return np.array([])

        synthetic_log_ISIs = []
        ctime = lead_time
        while True:
            synthetic_log_ISI = np.random.choice(trial_log_ISIs)
            ctime += 10**synthetic_log_ISI
            if ctime <= trial_length:
                synthetic_log_ISIs.append(synthetic_log_ISI)
            else:
                break
        
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
        x = np.linspace(-2, 20, 100)
        
        fftkde = FFTKDE(bw=bw, kernel='gaussian').fit(log_ISIs, weights=None)
        # min_fftkde_data = np.min(fftkde.data, axis=0)
        # max_fftkde_data = np.max(fftkde.data, axis=0)
        
        # x = np.linspace(int(min_fftkde_data) - 1, int(max_fftkde_data) + 1, 100)

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
        X : array-like, shape (num_trials, len_of_isi)
            Log ISIs for each trial.
        y : array-like, shape (num_trials,)
            Condition for each trial.
        synthetic : bool, default False
            Whether synthetic log ISI data should be constructed and used to
            fit the model.
        start_end_times : array-like, shape (num_trials, 2), optional
            The start and end times for each trial, to be used when
            generating synthetic log ISIs. This arg is required if 
            synthetic = True. 
        """ 

        decoding_conditions = self._create_condition_subset_mapping(X, y)

        if len(X) < self.min_ISIs:
            print("Insufficient LogISIs data for analysis.")
            return
        
        log_ISIs_concat = np.concatenate(X) # Flattens it to one array of ISIs

        if len(log_ISIs_concat) < self.min_ISIs:
            print("Insufficient LogISIs data after flattening for analysis.")
            return 

        # TODO: add error handling: if KDE throws data out of range error 
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
                    trial_log_ISIs, 
                    trial_start,
                    trial_end)
                synthetic_log_ISIs.append(trial_synthetic_log_ISIs)

            X = np.array(synthetic_log_ISIs,dtype='object')

        # Handle individual conditions
        num_ISIs_per_trial = [len(l) for l in X]
        total_empty_ISI_trials = np.sum(np.equal(num_ISIs_per_trial,0))

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
        X : array-like, shape (num_trials, len_of_isi)
            Log ISIs for each trial.
        y : array-like, shape (num_trials,)
            Condition for each trial.
        windows : array-like, shape (num_windows, 2)
            The start and end times (time-locked to trial starts) of each
            window to compute PDFs for. 
        synthetic : bool, default False
            Whether synthetic log ISI data should be constructed and used to
            fit the model.
        """ 

        condition_to_X = self._create_condition_subset_mapping(X, y)

        window_pdfs = {}
        window_inv_cdfs = {}

        for window_idx, (w_start, w_end) in enumerate(windows):
            window_ISIs_concat = []

            for trial in X:
                if window_idx < len(trial):
                    window_ISIs_concat.extend(trial[window_idx])
                
            window_ISIs_concat = np.array(window_ISIs_concat)

            if len(window_ISIs_concat) < self.min_ISIs:
                # print("Insufficient LogISIs data for analysis.")
                continue

            # TODO: add error handling: if KDE throws data out of range error 
            f,inv_f = self.estimate_ISI_distribution(window_ISIs_concat, self.bw)

            window_pdfs[(w_start, w_end)] = f
            window_inv_cdfs[(w_start, w_end)] = inv_f

        self.model.set_all(pdf=window_pdfs, inv_cdf=window_inv_cdfs)

        # Handle synthetic log ISI generation
        if synthetic:
            synthetic_log_ISIs = []

            for trial in X:
                synthetic_trial_log_ISIs = []

                for window_idx, window_ISIs in enumerate(trial):
                    w_start, w_end = windows[window_idx]

                    window_synthetic_log_ISIs = self.generate_synthetic_ISIs(
                        window_ISIs, 
                        w_start,
                        w_end)
                    synthetic_trial_log_ISIs.append(window_synthetic_log_ISIs)
                
                synthetic_log_ISIs.append(np.array(synthetic_trial_log_ISIs, dtype='object'))

            X = np.array(synthetic_log_ISIs,dtype='object')

        total_empty_ISI_trials = 0
        for trial in X:
            if np.all([len(window) == 0 for window in trial]):
                total_empty_ISI_trials += 1

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

                # TODO: add error handling: if KDE throws data out of range error 
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
        
        # If spike is not contained by any of the sliding windows:
        if best_window == (None, None):
            if len(windows) > 0:
                final_window = tuple(windows[-1])
                if spike - final_window[1] < 0.5:
                    best_window = final_window
                # print("Spike out of bounds, just selecting final window")
            # else: 
                # print("Spike out of bounds, making 0 PDF")

        return best_window

    def _predict_window_single_trial(
        self, 
        trial_log_ISIs: ArrayLike, 
        trial_final_spike: float,
        synthetic : bool = False,
        display_trial: bool = False,
        best_window_metric: str = 'center',
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
        trial_final_spike : float
            The final spike in the trial. Used to select which sliding window's
            PDF should be used in predictions.
        synthetic : bool, default False
            Whether synthetic spike train data should be constructed and used in
            model prediction.
        display_trial : bool, default False
            Whether to display a figure showing the prediction of the model 
            on the single trial. 
        best_window_metric : str
            The metric to use to select the best sliding window for the final spike.
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

        # Handle synthetic log ISI generation
        if synthetic:
            trial_log_ISIs = self.generate_synthetic_ISIs(trial_log_ISIs, trial_start, trial_end)

        # No ISIs in trial, make a guess based on priors
        if len(trial_log_ISIs) < 1:
            for cond in self.conditions:
                if self.model.conds[cond].prior_empty is None:
                    raise ValueError("Model's prior_empty uninitialized for %s condition." %cond)
                    
            max_cond = max(self.conditions, key=lambda cond: self.model.conds[cond].prior_empty)
            all_priors = {cond: self.model.conds[cond].prior_empty for cond in self.conditions}
            return max_cond, self.model.conds[max_cond].prior_empty, all_priors, True
        
        # ISIs were present in the trial, predict based on normalized probabilities
        else:
            # Set up probabilities with initial values equal to priors
            probabilities = {cond: np.full(len(trial_log_ISIs) + 1, np.nan)
                            for cond in self.conditions}
        
            for cond in self.conditions:
                if self.model.conds[cond].prior_0 is None:
                    raise ValueError("Model's prior_0 uninitialized for %s condition." %cond)
                    
                if self.model.conds[cond].pdf is None:
                    raise ValueError("Model's pdf uninitialized for %s condition." %cond)

                windows = np.sort(list(self.model.conds[cond].pdf.keys()))
                best_window = self._spike_to_window(trial_final_spike, windows, best_window_metric)
                best_pdf = None
                if best_window == (None, None):
                    best_pdf = lambda x : [0 for i in range(len(x))]
                else: 
                    best_pdf = self.model.conds[cond].pdf[best_window]

                # Calculate cumulative log-likelihood ratio (LLR) after each LogISI
                probabilities[cond] = np.cumsum(
                    np.log10(np.concatenate(([self.model.conds[cond].prior_0],
                                            best_pdf(trial_log_ISIs)))))

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

    def _predict_single_trial(
        self, 
        trial_log_ISIs: ArrayLike, 
        synthetic : bool = False,
        display_trial: bool = False,
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
            trial_log_ISIs = self.generate_synthetic_ISIs(trial_log_ISIs, trial_start, trial_end)

        # No ISIs in trial, make a guess based on priors
        if len(trial_log_ISIs) < 1:
            for cond in self.conditions:
                if self.model.conds[cond].prior_empty is None:
                    raise ValueError("Model's prior_empty uninitialized for %s condition." %cond)
                    
            max_cond = max(self.conditions, key=lambda cond: self.model.conds[cond].prior_empty)
            all_priors = {cond: self.model.conds[cond].prior_empty for cond in self.conditions}
            return max_cond, self.model.conds[max_cond].prior_empty, all_priors, True
        
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
        for trial_log_ISIs in X:
            pred_condition, _, _, _ = self._predict_single_trial(trial_log_ISIs)
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
        for trial_log_ISIs in X:
            _, _, trial_probs, _ = self._predict_single_trial(trial_log_ISIs)
            probabilities.append([trial_probs[cond] for cond in self.conditions])
        return pd.DataFrame(probabilities, columns=self.conditions)
    
    def display_single_trial(
        self, 
        trial_log_ISIs: ArrayLike, 
        file_path: Optional[str] = None):
        """
        Displays and (optionally) saves a figure showing the prediction of the 
        model on a single trial. 

        Parameters
        ----------
        trial_log_ISIs : array-like, shape (num ISIs, )
            The log ISIs for a single trial. 
        file_path : str
            Path to store generated figure at.
        """
        self._predict_single_trial(
            trial_log_ISIs, 
            display_trial=True,
            file_path=file_path)

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
        num_correct = 0
        num_empty_ISIs = 0
        num_preds = 0

        if synthetic and start_end_times is None:
            raise ValueError("Trial start and end times must be passed in to generate synthetic log ISI data")
        
        for idx, trial in enumerate(test_X):
            trial_start, trial_end = None, None
            if synthetic:
                trial_start, trial_end = start_end_times[idx]

            cond, _, _, empty_ISIs = self._predict_single_trial(
                trial, 
                synthetic=synthetic, 
                trial_start=trial_start, 
                trial_end=trial_end)
            
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

    def calculate_circular_MSE(self, 
        test_X: ArrayLike, 
        test_y: ArrayLike, 
        max_cond : float,
        synthetic: bool = False,
        start_end_times: Optional[ArrayLike] = None) -> tuple[float, float]:

        num_empty_ISIs = 0
        num_preds = 0
        mse = 0
        rand_mse = 0
        pred_y = []

        if synthetic and start_end_times is None:
            raise ValueError("Trial start and end times must be passed in to generate synthetic log ISI data")

        for idx, trial in enumerate(test_X):
            trial_start, trial_end = None, None
            if synthetic:
                trial_start, trial_end = start_end_times[idx]

            cond, _, _, empty_ISIs = self._predict_single_trial(
                trial, 
                synthetic=synthetic, 
                trial_start=trial_start, 
                trial_end=trial_end)
            
            if cond is None:
                continue

            if empty_ISIs:
                num_empty_ISIs += 1
            num_preds += 1
            pred_y.append(cond)
        
            labels = [i for i in range(1, 9)]
            rand_pred = random.choice(labels)
            pred = float(cond)
            actual = float(test_y[idx])

            circ_dist = min(abs(max_cond - pred), abs(0 - pred)) + min(abs(max_cond - actual), abs(0 - actual))
            rand_circ_dist = min(abs(max_cond - rand_pred), abs(0 - rand_pred)) + min(abs(max_cond - actual), abs(0 - actual))
            
            err = min(abs(pred - actual), circ_dist)
            mse += err 
            rand_err = min(abs(rand_pred - actual), rand_circ_dist)
            rand_mse += rand_err

        if num_preds == 0:
            return np.nan, np.nan, np.nan, np.nan
        else:
            return (mse/num_preds), (rand_mse/num_preds), (num_empty_ISIs/num_preds), pred_y
    
    def calculate_window_accuracy(
        self, 
        test_X: ArrayLike, 
        test_y: ArrayLike, 
        test_final_spikes: ArrayLike,
        synthetic: bool = False,
        start_end_times: Optional[ArrayLike] = None) -> tuple[float, float]:
        """
        Calculates the accuracy of the decoder on unseen test trial data using 
        sliding windows.

        Parameters
        ----------
        test_X : array-like, shape (num trials, num ISIs per trial)
            Log ISIs for each trial.
        test_y : array-like, shape (num trials,)
            Condition of each trial.
        trial_final_spikes : float
            The final spike in each trial. Used to select which sliding window's
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

        num_correct = 0
        num_empty_ISIs = 0
        num_preds = 0

        if synthetic and start_end_times is None:
            raise ValueError("Trial start and end times must be passed in to generate synthetic log ISI data")
        
        for idx, trial in enumerate(test_X):
            trial_start, trial_end = None, None
            if synthetic:
                trial_start, trial_end = start_end_times[idx]

            cond, _, _, empty_ISIs = self._predict_window_single_trial(
                trial, 
                test_final_spikes[idx],
                synthetic=synthetic, 
                trial_start=trial_start, 
                trial_end=trial_end)
            
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
    
    def calculate_window_circular_MSE(self, 
        test_X: ArrayLike, 
        test_y: ArrayLike, 
        test_final_spikes: ArrayLike,
        max_cond : float,
        synthetic: bool = False,
        start_end_times: Optional[ArrayLike] = None) -> tuple[float, float]:

        num_empty_ISIs = 0
        num_preds = 0
        mse = 0
        rand_mse = 0
        pred_y = []

        if synthetic and start_end_times is None:
            raise ValueError("Trial start and end times must be passed in to generate synthetic log ISI data")

        for idx, trial in enumerate(test_X):
            trial_start, trial_end = None, None
            if synthetic:
                trial_start, trial_end = start_end_times[idx]

            cond, _, _, empty_ISIs = self._predict_window_single_trial(
                trial, 
                test_final_spikes[idx],
                synthetic=synthetic, 
                trial_start=trial_start, 
                trial_end=trial_end)
            
            if cond is None:
                continue

            if empty_ISIs:
                num_empty_ISIs += 1
            num_preds += 1
            pred_y.append(cond)
        
            labels = [i for i in range(1, 9)]
            rand_pred = random.choice(labels)
            pred = float(cond)
            actual = float(test_y[idx])

            circ_dist = min(abs(max_cond - pred), abs(0 - pred)) + min(abs(max_cond - actual), abs(0 - actual))
            rand_circ_dist = min(abs(max_cond - rand_pred), abs(0 - rand_pred)) + min(abs(max_cond - actual), abs(0 - actual))
            
            err = min(abs(pred - actual), circ_dist)
            mse += err 
            rand_err = min(abs(rand_pred - actual), rand_circ_dist)
            rand_mse += rand_err

        if num_preds == 0:
            return np.nan, np.nan, np.nan, np.nan
        else:
            return (mse/num_preds), (rand_mse/num_preds), (num_empty_ISIs/num_preds), pred_y
    
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