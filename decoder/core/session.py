import os
import numpy as np
import pynapple as nap
from typing import Union, Optional
from numpy.typing import ArrayLike
from decoder.core.unique_interval_set import UniqueIntervalSet

class Session:
    """ Preprocesses spike train data and computes log ISIs."""
    
    def __init__(
        self, 
        spike_times: Union[ArrayLike, nap.Ts], 
        interval_sets: dict[str, nap.IntervalSet], 
        event_sets: dict[str, nap.Tsd]):
        """
        Parameters
        ----------
        spike_times : array-like or nap.Ts
            Spike times for the session (one dimensional) 
        interval_sets : dict[str, nap.IntervalSet]
            Dict of interval sets for the session. The keys are the name of 
            each interval set. 
        event_sets : dict[str, nap.Tsd]: 
            Dict of event sets for the session. The keys are the name of each
            event set.
        """

        if not isinstance(spike_times, nap.Ts):
            self.spike_train = nap.Ts(t=spike_times)
        else:
            self.spike_train = spike_times
        
        if not isinstance(interval_sets, dict):
            raise TypeError("interval_sets must be a dict.")
        if not all(isinstance(key, str) for key in interval_sets.keys()):
            raise TypeError("All keys in interval_sets dict must be strings.")
        if not all(isinstance(val, nap.IntervalSet) for val in interval_sets.values()):
            raise TypeError("All values in interval_sets dict must be nap.IntervalSet.")
        
        if not isinstance(event_sets, dict):
            raise TypeError("event_sets must be a dict.")
        if not all(isinstance(key, str) for key in event_sets.keys()):
            raise TypeError("All keys in event_sets dict must be strings.")
        if not all(isinstance(val, nap.Tsd) for val in event_sets.values()):
            raise TypeError("All values in event_sets dict must be nap.Tsd.")

        self.interval_sets = interval_sets
        self.event_sets = event_sets

        self.interval_to_spikes = {} # Maps iset_name -> start, end -> spikes Ts
        self.locked_interval_to_spikes = {} # Maps (event_name, iset_name) -> start, end -> spikes Ts
        self.interval_to_padded_spikes = {} # Maps iset_name -> start, end -> padded spikes Ts
        self.locked_interval_to_padded_spikes = {} # Maps (event_name, iset_name) -> start, end -> padded spikes Ts

        self.iset_to_log_ISIs = {} # Maps iset_name -> np 2D array of log ISIs
        self.locked_iset_to_log_ISIs = {} # Maps (event_name, iset_name) -> np array of log ISIs

        self.iset_to_windows = {} # Maps iset_name -> np array of windows
        self.iset_to_windowed_log_ISIs = {} # Maps iset_name -> np array of windowed log ISIs

    def slice_spikes_by_intervals(self, iset_name: str):
        """
        Restricts spike data to a specified interval set. 
        Maps each interval in the interval set to the spikes contained within 
        that interval. If interval set is a padded UniqueIntervalSet, this 
        function also maps each interval to the spikes contained within the 
        padded interval.

        Parameters
        ----------
        iset_name : str
            The name of the IntervalSet to restrict the spikes to.
        """
        if iset_name not in self.interval_sets:
            raise KeyError("Interval set name does not exist")

        interval_set = self.interval_sets[iset_name]
        
        # Return if result is already computed and stored.
        if iset_name in self.interval_to_spikes:
            return

        self.interval_to_spikes[iset_name] = {}

        if isinstance(interval_set, UniqueIntervalSet):
            if interval_set.padded_start is not None or interval_set.padded_end is not None:
                self.interval_to_padded_spikes[iset_name] = {}
        
        # Iterate through the intervals and map spikes.
        for idx in range(len(interval_set.start)):
            start, end = interval_set.start[idx], interval_set.end[idx]
            interval_spikes = self.spike_train.get(start=start, end=end)
            self.interval_to_spikes[iset_name][(start, end)] = interval_spikes

            # Map padded spikes to interval if needed.
            if isinstance(interval_set, UniqueIntervalSet):
                if interval_set.padded_start is not None or interval_set.padded_end is not None:
                    padded_start = interval_set.padded_start[idx] if interval_set.padded_start is not None else start
                    padded_end = interval_set.padded_end[idx] if interval_set.padded_end is not None else end

                    padded_interval_spikes = self.spike_train.get(start=padded_start, end=padded_end)
                    self.interval_to_padded_spikes[iset_name][(start, end)] = padded_interval_spikes

    def compute_log_ISIs(
        self, 
        iset_name: str, 
        lock_point: Optional[str] = None, 
        scaling_factor: Union[int, float] = 1000):
        """
        Computes log ISIs for a specific interval set.
        If a lock point is passed in, the function first time-locks the spike 
        train to that lock point in each interval before computing log ISIs.

        Parameters
        ----------
        iset_name : str
            Name of the interval set to compute log ISIs for.
        lock_point : str, optional
            Either the name of the event set that the spike trains should be 
            time-locked to, or 'start'/'end' to time-lock to the start/end of 
            each interval.
        scaling_factor : int, optional
            Factor to scale the spike times. Defaults to 1000.

        Returns
        -------
        log_ISIs : ndarray, shape (num trials, num ISIs per trial)
            2D array containing log ISIs for the specified interval set. 
        """
        log_ISIs_agg = []
        interval_set = self.interval_sets[iset_name]

        self.slice_spikes_by_intervals(iset_name=iset_name)

        if lock_point is not None:
            if lock_point in ['start', 'end']:
                self.time_lock_to_interval(iset_name=iset_name, lock_point=lock_point)
            else:
                self.time_lock_to_event(eset_name=lock_point, iset_name=iset_name)

        for start, end in zip(interval_set.start, interval_set.end):
            spikes = None
            if lock_point is not None:
                spikes = self.locked_interval_to_spikes[(lock_point, iset_name)][(start, end)]
            else:
                spikes = self.interval_to_spikes[iset_name][(start, end)]
            spikes = spikes.t
            
            # Scaling the spikes using numpy's vectorized operation
            scaled_spikes = spikes * scaling_factor

            interval_ISIs = np.diff(scaled_spikes)
            
            # Condition to check for non-empty interval_ISIs before applying log
            # Only apply log to entries that are greater than zero 
            interval_log_ISIs = np.log10(interval_ISIs, out=np.zeros(interval_ISIs.shape), where=(interval_ISIs > 0)) if interval_ISIs.size != 0 else np.array([])
            log_ISIs_agg.append(interval_log_ISIs)
            
        # Store log_ISIs for the current iset_name in the dictionary
        if lock_point is not None:
            self.locked_iset_to_log_ISIs[(lock_point, iset_name)] = np.array(log_ISIs_agg, dtype='object')
            return self.locked_iset_to_log_ISIs[(lock_point, iset_name)]

        self.iset_to_log_ISIs[iset_name] = np.array(log_ISIs_agg, dtype='object')
        return self.iset_to_log_ISIs[iset_name]

    def _final_spikes_in_trials(
        self,
        iset_name: str, 
        lock_point: str):
        """
        Returns the (time-locked) "final" spike for each log ISI in each interval in 
        the interval set.

        Parameters
        ----------
        iset_name : str
            Name of the interval set to find final spikes for.
        lock_point : str
            Lock point that the interval set is already time-locked to.

        Returns
        -------
        final_spikes_agg : ndarray, shape (num trials, )
            Array containing the time-locked final spikes for each log ISI in each trial.
        """
        final_spikes_agg = []
        interval_set = self.interval_sets[iset_name]

        for start, end in zip(interval_set.start, interval_set.end):
            spikes = self.locked_interval_to_spikes[(lock_point, iset_name)][(start, end)]
            spikes = spikes.t
            final_spikes = spikes[1:]
            final_spikes_agg.append(final_spikes)
        return np.array(final_spikes_agg, dtype='object')

    def compute_windowed_log_ISIs(
        self, 
        iset_name: str,
        window_len: float = 1.0, 
        step: float = 0.1,
        scaling_factor: Union[int, float] = 1000):
        """
        Computes log ISIs for a specific interval set using sliding windows.
        The function first time-locks the spike train to the start of each
        interval, then generates sliding windows across each interval 
        and computes log ISIs for each sliding window.

        Parameters
        ----------
        iset_name : str
            Name of the interval set to compute log ISIs for.
        window_len : float (default = 1.0)
            The length of each sliding window (in seconds).
        step : float (default = 0.1)
            The step size between sliding windows (in seconds).
        scaling_factor : int or float (default = 1000)
            Factor to scale the spike times.

        Returns
        -------
        windowed_log_ISIs : ndarray, shape (num trials, num sliding windows in trial, num ISIs per window)
            Contains log ISIs for each sliding window in each interval.
        windows : ndarray, shape (num sliding windows, )
            The sliding windows used across all intervals in the interval set. 
            Note that each interval may not use all of the sliding windows in 
            this array, depending on its duration.
        final_spikes : ndarray, shape (num trials, )
            The final (time-locked) spikes for each log ISI in each trial.
        """
        interval_set = self.interval_sets[iset_name]
        lock_point = 'start'

        self.slice_spikes_by_intervals(iset_name=iset_name)

        self.time_lock_to_interval(iset_name=iset_name, lock_point=lock_point)

        final_spikes = self._final_spikes_in_trials(iset_name=iset_name, lock_point=lock_point)
        
        windows_agg = []
        max_num_windows = 0
        windowed_log_ISIs_agg = []

        # If interval set is padded, compute sliding windows to cover entire padded interval.
        if (lock_point, iset_name) in self.locked_interval_to_padded_spikes:
            for idx in range(len(interval_set.start)):
                start, end = interval_set.start[idx], interval_set.end[idx]
                spikes = self.locked_interval_to_padded_spikes[(lock_point, iset_name)][(start, end)]

                if interval_set.padded_start is not None:
                    padded_start = interval_set.padded_start[idx]
                else:
                    padded_start = start
                if interval_set.padded_end is not None:
                    padded_end = interval_set.padded_end[idx]
                else:
                    padded_end = end

                first_window_center = padded_start - start + (window_len / 2.) # - start since spikes are time-locked to start
                last_window_center = padded_end - start - (window_len / 2.) # - start since spikes are time-locked to start
                windows = [(i - window_len / 2., i + window_len / 2.) for i in np.arange(first_window_center, last_window_center + step, step)]

                if len(windows) > max_num_windows:
                    max_num_windows = len(windows)
                    windows_agg = windows

                all_interval_spikes = spikes.t
                trial_windowed_log_ISIs = []

                for w_start, w_end in windows:
                    window_spikes = spikes.get(w_start, w_end)
                    window_spikes = window_spikes.t

                    # Add spike right before window to include ISI that ends in window
                    if len(window_spikes) > 0:
                        first_spike = window_spikes[0]
                        first_spike_idx = np.where(all_interval_spikes == first_spike)[0][0]

                        if first_spike_idx > 0:
                            prev_spike = all_interval_spikes[first_spike_idx - 1]
                            window_spikes = np.insert(window_spikes, 0, prev_spike)

                    scaled_spikes = window_spikes * scaling_factor

                    interval_ISIs = np.diff(scaled_spikes)
                    
                    # Condition to check for non-empty interval_ISIs before applying log
                    # Only apply log to entries that are greater than zero 
                    interval_log_ISIs = np.log10(interval_ISIs, out=np.zeros(interval_ISIs.shape), where=(interval_ISIs > 0)) if interval_ISIs.size != 0 else np.array([])
                    
                    trial_windowed_log_ISIs.append(interval_log_ISIs)

                windowed_log_ISIs_agg.append(trial_windowed_log_ISIs)
        else:
            # Interval set is not padded. Compute sliding windows that are contained entirely within interval.
            for start, end in zip(interval_set.start, interval_set.end):
                spikes = self.locked_interval_to_spikes[(lock_point, iset_name)][(start, end)]

                first_window_center = window_len / 2. # - start since spikes are time-locked to start
                last_window_center = end - start - (window_len / 2.) # - start since spikes are time-locked to start
                windows = [(i - window_len / 2., i + window_len / 2.) for i in np.arange(first_window_center, last_window_center + step, step)]

                if len(windows) > max_num_windows:
                    max_num_windows = len(windows)
                    windows_agg = windows

                all_interval_spikes = spikes.t
                trial_windowed_log_ISIs = []

                for w_start, w_end in windows:
                    window_spikes = spikes.get(w_start, w_end)
                    window_spikes = window_spikes.t

                    # Add spike right before window to include ISI that ends in window
                    if len(window_spikes) > 0:
                        first_spike = window_spikes[0]
                        first_spike_idx = np.where(all_interval_spikes == first_spike)[0][0]

                        if first_spike_idx > 0:
                            prev_spike = all_interval_spikes[first_spike_idx - 1]
                            window_spikes = np.insert(window_spikes, 0, prev_spike)

                    scaled_spikes = window_spikes * scaling_factor

                    interval_ISIs = np.diff(scaled_spikes)
                    
                    # Condition to check for non-empty interval_ISIs before applying log
                    # Only apply log to entries that are greater than zero 
                    interval_log_ISIs = np.log10(interval_ISIs, out=np.zeros(interval_ISIs.shape), where=(interval_ISIs > 0)) if interval_ISIs.size != 0 else np.array([])
                    
                    trial_windowed_log_ISIs.append(interval_log_ISIs)

                windowed_log_ISIs_agg.append(trial_windowed_log_ISIs)
        
        self.iset_to_windows[iset_name] = np.array(windows_agg)
        self.iset_to_windowed_log_ISIs[iset_name] = np.array(windowed_log_ISIs_agg, dtype='object')

        return self.iset_to_windowed_log_ISIs[iset_name], self.iset_to_windows[iset_name], final_spikes


    def time_lock_to_interval(self, iset_name: str, lock_point: str):
        """
        Time-locks spike trains in a given interval set to the start or end of 
        their interval.
        
        This function treats the start/end of the interval as the new 'zero', 
        and shifts the spike times accordingly. It then stores the time-locked
        spike trains in self.locked_interval_to_spikes.
        
        If the interval set is padded, this function also time-locks the padded
        spike trains to the start/end of their interval, and stores them in
        self.locked_interval_to_padded_spikes.

        Parameters
        ----------
        iset_name : str
            Name of the interval set for which spike trains should be time-locked.
        lock_point : str
            Either 'start' or 'end', indicating whether spikes should be 
            time-locked to the start or the end of their interval. 
        """
        if iset_name not in self.interval_sets:
            raise KeyError("Interval set name does not exist")
        if iset_name not in self.interval_to_spikes:
            raise KeyError("Spikes have not been mapped to this interval set. Run slice_spikes_by_intervals first.")
        
        if lock_point not in ['start', 'end']:
            raise ValueError("lock_point should be either 'start' or 'end'")

        # Return if result is already computed and stored.
        if (lock_point, iset_name) in self.locked_interval_to_spikes:
            return

        for (start, end), spikes in self.interval_to_spikes[iset_name].items():
            if lock_point == 'start':
                time_adjustment = start
            else:
                time_adjustment = end
                
            # Adjust spike times based on the lock point
            locked_spikes = [spike - time_adjustment for spike in spikes.t]
            
            # Store in the mapped structure
            key = (lock_point, iset_name)
            if key not in self.locked_interval_to_spikes:
                self.locked_interval_to_spikes[key] = {}
            self.locked_interval_to_spikes[key][(start, end)] = nap.Ts(t=np.array(locked_spikes))
        
        # If interval set is padded, also time-locks the padded spike trains.
        if iset_name in self.interval_to_padded_spikes:
            for (start, end), spikes in self.interval_to_padded_spikes[iset_name].items():
                if lock_point == 'start':
                    time_adjustment = start
                else:
                    time_adjustment = end
                    
                # Adjust spike times based on the lock point
                locked_spikes = [spike - time_adjustment for spike in spikes.t]
                
                # Store in the mapped structure
                key = (lock_point, iset_name)
                if key not in self.locked_interval_to_padded_spikes:
                    self.locked_interval_to_padded_spikes[key] = {}
                self.locked_interval_to_padded_spikes[key][(start, end)] = nap.Ts(t=np.array(locked_spikes))


    def _match_event_to_interval(self, event_set: nap.Tsd, iset_name: str) -> dict:
        """
        Matches each interval in the interval set to the corresponding event
        occurring in that interval. There should be exactly one event
        corresponding to each interval in the interval set.

        Parameters
        ----------
        event_set : nap.Tsd
            Event set to be matched with intervals.
        iset_name : str
            Name of the interval set being matched.

        Returns
        -------
        matched_events : dict
            Maps each interval to the timestamp of its corresponding event.

        """
        matched_events = {}
        for start, end in self.interval_to_spikes[iset_name].keys():
            events_in_interval = event_set.get(start=start, end=end)

            if len(events_in_interval) != 1:
                raise ValueError(f"Interval ({start}, {end}) does not match exactly one event in event_set.")
            matched_events[(start, end)] = events_in_interval.t[0]

        return matched_events

    def time_lock_to_event(self, eset_name: str, iset_name: str):
        """
        Time-locks spike trains in a given interval set to the corresponding
        event that occurs in their interval.

        This function treats the event occuring in the interval as the new 
        'zero', and shifts the spike times accordingly. It then stores the 
        time-locked spike trains in self.locked_interval_to_spikes.

        If the interval set is padded, this function also time-locks the padded
        spike trains to the event time, and stores them in
        self.locked_interval_to_padded_spikes.

        Parameters
        ----------
        eset_name : str 
            Name of the event set to time-lock spikes with.
        iset_name : str 
            Name of the interval set for which spike trains should be time-locked.
        """
        if iset_name not in self.interval_sets:
            raise KeyError("Interval set name does not exist")
        if iset_name not in self.interval_to_spikes:
            raise KeyError("Spikes have not been mapped to this interval set. Run slice_spikes_by_intervals first.")
        
        if eset_name not in self.event_sets:
            raise KeyError("Event set name passed in as lock point does not exist")

        # Return if result is already computed and stored.
        if (eset_name, iset_name) in self.locked_interval_to_spikes:
            return

        event_set = self.event_sets[eset_name]
        matched_events = self._match_event_to_interval(event_set, iset_name)

        for (start, end), spikes in self.interval_to_spikes[iset_name].items():
            event_time = matched_events[(start, end)]
            locked_spikes = [spike - event_time for spike in spikes.t]

            key = (eset_name, iset_name)
            if key not in self.locked_interval_to_spikes:
                self.locked_interval_to_spikes[key] = {}
            self.locked_interval_to_spikes[key][(start, end)] = nap.Ts(t=np.array(locked_spikes))

        # If interval set is padded, also time-locks the padded spike trains.
        if iset_name in self.interval_to_padded_spikes:
            for (start, end), spikes in self.interval_to_padded_spikes[iset_name].items():
                event_time = matched_events[(start, end)]
                locked_spikes = [spike - event_time for spike in spikes.t]

                key = (eset_name, iset_name)
                if key not in self.locked_interval_to_padded_spikes:
                    self.locked_interval_to_padded_spikes[key] = {}
                self.locked_interval_to_padded_spikes[key][(start, end)] = nap.Ts(t=np.array(locked_spikes))

    def save_log_ISIs(self, iset_name: str, file_path: str, lock_point: Optional[str] = None):
        """
        Saves computed log ISIs for a specific interval set to a .npz file. 
        If a lock point is provided, it first time locks the interval set before
        computing log ISIs.

        Parameters
        ----------
        iset_name : str
            Name of the interval set to save log ISIs for.
        eset_name : str, optional
            Name of the lock point (an event set, or the start/end of the 
            interval) to time lock to.
        file_path : str
            Path to save the log ISIs .npz file at.

        Loading Example:
            with np.load(file_path, allow_pickle=True) as data:
                log_ISIs = data['log_ISIs']
        """

        if lock_point is not None:
            if (lock_point, iset_name) not in self.locked_iset_to_log_ISIs:
                raise KeyError("Invalid lock point, or log ISIs have not been computed for this interval and lock point")
        elif iset_name not in self.iset_to_log_ISIs:
            raise KeyError("Interval set does not exist, or log ISIs have not been computed for this interval set")

        self._validate_file_path(file_path)
        if lock_point is not None:
            locked_log_ISIs = self.locked_iset_to_log_ISIs[(lock_point, iset_name)]
            np.savez(file_path, locked_log_ISIs=locked_log_ISIs)
        else:
            log_ISIs = self.iset_to_log_ISIs[iset_name]
            np.savez(file_path, log_ISIs=log_ISIs)


    @staticmethod
    def _validate_file_path(file_path: str):
        """
        Validates the provided file path.

        Parameters
        ----------
        file_path : str
            File path to validate.
        """
        if not isinstance(file_path, str):
            raise RuntimeError("File path should be a string.")
        if os.path.isdir(file_path):
            raise RuntimeError(f"{file_path} is a directory.")
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            raise RuntimeError(f"Path {directory} does not exist.")

