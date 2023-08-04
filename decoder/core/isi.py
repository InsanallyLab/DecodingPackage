
import numpy as np 
from KDEpy import FFTKDE 
from scipy.interpolate import interp1d 

class ISI: 
    def __init__(self, loader, spike_train) -> None:
        self.loader =loader 
        self.spike_train = spike_train 
        self.cached_log_isis = {}

    def get_log_isis(self, interval, trials = None):
        """
        Computes and caches the log inter-spike intervals (LogISIs) for the given interval and trials.
        Args:
            interval (Interval): The time interval for which to compute the LogISIs.
            trials (list or None): A list of trial indices to consider. Default is None, which
                means all trials in the session will be considered.

        Returns:
            numpy.ndarray: An array containing LogISIs for each trial in the specified interval.

        Raises:
            ValueError: If `trials` contains invalid trial indices.
        """ 

        if (trials is None): 
            trials = range(self.loader.meta.length_in_trials) 

        # Check if LogISIs are already cached for the given interval and cluster
        cache_key = (trials, interval) 
        if cache_key in self.cached_log_ISIs:
            return self.cached_log_ISIs[cache_key]

        ISIs = []
        for trial in range(self.loader.meta.length_in_trials):
            starttime, endtime = interval._ToTimestamp(self.session_loader.trials, trial)
            spiketimes = self.getSpikeTimes(starttime=starttime, endtime=endtime)
            spiketimes = spiketimes * 1000 / self.loader.meta.fs

            ISIs.append(np.diff(spiketimes))

        LogISIs = np.array([np.log10(tISIs) for tISIs in ISIs], dtype='object')

        # Cache the LogISIs for future use
        self.cached_log_ISIs[cache_key] = LogISIs
        return LogISIs

    @staticmethod
    def estimate_isi_distribution(log_isis, bandwidth):
        """
        Estimate the Inter-Spike Interval (ISI) Distribution using Kernel Density Estimation (KDE).

        Args:
            log_isis (numpy.ndarray): An array of log-transformed Inter-Spike Interval (ISI) values.
            bandwidth (float): The bandwidth parameter for the KDE.

        Returns:
            tuple: A tuple containing two functions:
                - A function that interpolates the estimated probability density function (PDF).
                - A function to generate an inverse Cumulative Distribution Function (CDF) for sampling.
        """
        x = np.linspace(-2, 6, 100)
        y = FFTKDE(bw=bandwidth, kernel='gaussian').fit(log_isis, weights=None).evaluate(x)

        # Use scipy to interpolate and evaluate on an arbitrary grid
        pdf_interpolator = interp1d(x, y, kind='linear', assume_sorted=True)

        # Generate an inverse CDF for sampling
        norm_y = np.cumsum(y) / np.sum(y)
        norm_y[0] = 0
        norm_y[-1] = 1
        cdf_inverse_interpolator = interp1d(norm_y, x, kind='linear', assume_sorted=True)

        return pdf_interpolator, cdf_inverse_interpolator