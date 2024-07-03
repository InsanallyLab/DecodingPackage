import os
import numpy as np
import pynapple as nap
from typing import Union, Optional
from numpy.typing import ArrayLike, NDArray

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
            Spike times for the session. 
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
        
        # CHANGED: dict of IntervalSets instead of list of IntervalSets
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
        # CHANGED: dict of Tsd instead of list of EventSets 
        self.event_sets = event_sets

        self.iset_to_spikes = {} # Maps interval_name -> spikes Ts
        self.interval_to_spikes = {} # Maps interval_name -> start, end -> spikes Ts
        self.iset_to_log_ISIs = {} # Maps interval_name -> np array of log ISIs

        self.locked_iset_to_spikes = {} # Maps (event_name, interval_name) -> spikes Ts
        self.locked_interval_to_spikes = {} # Maps (event_name, interval_name) -> start, end -> spikes Ts
        self.locked_iset_to_log_ISIs = {} # Maps (event_name, interval_name) -> np array of log ISIs

    def slice_spikes_by_intervals(self, iset_name: str):
        """
        Restricts spike data to a specified interval and caches the result.

        This function restricts spike data to a given interval defined by the 
        provided interval name. If the results have been previously computed, 
        they are retrieved from cache to speed up the process.

        Parameters
        ----------
        iset_name : str
            The name of the IntervalSet to restrict the spikes to.
        """
        if iset_name not in self.interval_sets:
            raise KeyError("Interval set name does not exist")

        # Fetch the desired interval set (padded if padding has been added to the object).
        interval_set = self.interval_sets[iset_name]
        
        # Return if result is already computed and stored.
        if iset_name in self.iset_to_spikes:
            return
        
        # Restrict the spike train to the interval.
        restricted_spikes = self.spike_train.restrict(interval_set)

        # Initialize a storage for mapped spikes if not present.
        if iset_name not in self.interval_to_spikes: 
            self.interval_to_spikes[iset_name] = {}
        
        # Iterate through the intervals and map spikes.
        for start, end in zip(interval_set.start, interval_set.end):
            # CHANGED: for spike train as Ts
            interval_spikes = restricted_spikes.get(start=start, end=end)

            self.interval_to_spikes[iset_name][(start, end)] = interval_spikes
        
        # Cache the result for future use.
        self.iset_to_spikes[iset_name] = restricted_spikes

    def compute_log_ISIs(
        self, 
        iset_name: str, 
        lock_point: Optional[str] = None, 
        scaling_factor: Union[int, float] = 1000):
        """
        Computes log ISIs for a specific interval set and stores them.
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
        numpy.ndarray: 2D array containing logISIs for the specified interval 
        set. Shape is (num trials, num ISIs per trial)
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


    def time_lock_to_interval(self, iset_name: str, lock_point: str):
        """
        Time-locks spike trains in a given interval set to the start or end of 
        their interval.
        
        This function treats the start/end of the interval as the new 'zero', 
        and shifts the spike times accordingly. It then stores the time-locked
        spike trains in self.locked_iset_to_spikes and 
        self.locked_interval_to_spikes.
        
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
        if (lock_point, iset_name) in self.locked_iset_to_spikes:
            return

        locked_spikes_agg = []

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
            
            locked_spikes_agg.extend(locked_spikes)

        # Convert the accumulated spikes into a Ts object and store
        tsd_obj = nap.Ts(t=np.array(locked_spikes_agg))
        self.locked_iset_to_spikes[(lock_point, iset_name)] = tsd_obj


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
        dict: Maps each interval to the timestamp of its corresponding event.

        """
        matched_events = {}
        for start, end in self.interval_to_spikes[iset_name].keys():
            # CHANGED: Tsd instead of EventSet
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
        time-locked spike trains in self.locked_iset_to_spikes and 
        self.locked_interval_to_spikes.

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
        if (eset_name, iset_name) in self.locked_iset_to_spikes:
            return

        event_set = self.event_sets[eset_name]
        matched_events = self._match_event_to_interval(event_set, iset_name)
        locked_spikes_agg = []

        for (start, end), spikes in self.interval_to_spikes[iset_name].items():
            event_time = matched_events[(start, end)]
            locked_spikes = [spike - event_time for spike in spikes.t]

            key = (eset_name, iset_name)
            if key not in self.locked_interval_to_spikes:
                self.locked_interval_to_spikes[key] = {}
            self.locked_interval_to_spikes[key][(start, end)] = nap.Ts(t=np.array(locked_spikes))

            locked_spikes_agg.extend(locked_spikes)

        self.locked_iset_to_spikes[(eset_name, iset_name)] = nap.Ts(t=np.array(locked_spikes_agg))


    def save_spikes(self, iset_name: str, file_path: str, lock_point: Optional[str] = None):
        """
        Saves spike train for a specific interval set to a .npz file. 
        If a lock point is provided, it saves the time-locked spike train instead.

        Parameters
        ----------
        iset_name : str
            Name of the interval set to save a spike train for.
        eset_name : str, optional
            Name of the lock point (an event set, or the start/end of the 
            interval) to time lock to.
        file_path : str 
            Path to save the spike train .npz file at.

        Loading Example:
            spike_train = nap.load_file(file_path)
        """
        if lock_point is not None:
            if (lock_point, iset_name) not in self.locked_iset_to_spikes:
                raise KeyError("Invalid lock point, or spikes have not been mapped to this interval set and lock point")
        elif iset_name not in self.iset_to_spikes:
            raise KeyError("Interval set does not exist, or spikes have not been mapped to this interval set")

        self._validate_file_path(file_path)
        if lock_point is not None:
            self.locked_iset_to_spikes[(lock_point, iset_name)].save(file_path)
        else:
            self.iset_to_spikes[iset_name].save(file_path)


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

