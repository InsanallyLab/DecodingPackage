import os
from wsgiref import validate
import numpy as np
import pynapple as nap
import pandas as pd

class Session:

    def __init__(self, spike_times, interval_sets, event_sets, name=None):
        """Initialize the Session class.

        Args:
            spike_times (array-like): Spike times for SpikeTrain.
            interval_sets (array-like): List of interval sets for the session. 
            event_sets (array-like): List of EventSets for the session.
            name (str, optional): Session name. Defaults to None.
        """
        self.name = name

        if not isinstance(spike_times, nap.Ts):
        # CHANGED: spike_train is Ts instead of Tsd
            self.spike_train = nap.Ts(t=spike_times)
        else:
            self.spike_train = spike_times
            
        self.interval_sets = {iset.name: iset for iset in interval_sets}
        self.event_sets = {eset.name: eset for eset in event_sets}

        self.iset_to_spikes = {} #maps interval_name -> spikes Ts
        self.interval_to_spikes = {} #maps interval_name -> start,end -> spikes Ts
        self.iset_to_log_ISIs = {} #maps interval_name -> np array of log ISIs

        self.locked_iset_to_spikes = {} #maps (event_name, interval_name) -> spikes Ts
        self.locked_interval_to_spikes = {} #maps (event_name, interval_name) -> start,end -> spikes Ts
        self.locked_iset_to_log_ISIs = {} #maps (event_name, interval_name) -> np array of log ISIs

    def add_interval_set(self, interval_set, override=False):
        """Add an interval set to the session.

        Args:
            interval_set (UniqueIntervalSet): The interval set to add.
            override (bool, optional): Whether to override if an interval set with the same name exists. Defaults to False.

        Raises:
            Warning: Raises a warning if an interval set with the same name exists and override is False.
        """
        if interval_set.name in self.interval_sets and not override:
            raise Warning(f"Interval set with name {interval_set.name} already exists. Use 'override=True' to replace it.")
        elif interval_set.name in self.interval_sets and override:
            print(f"Warning: Overriding existing interval set with name {interval_set.name}")
            
        self.interval_sets[interval_set.name] = interval_set

    def add_event_set(self, event_set, override=False):
        """Add an event set to the session.

        Args:
            event_set (EventSet): The event set to add.
            override (bool, optional): Whether to override if an event set with the same name exists. Defaults to False.

        Raises:
            Warning: Raises a warning if an event set with the same name exists and override is False.
        """
        if event_set.name in self.event_sets and not override:
            raise Warning(f"Event set with name {event_set.name} already exists. Use 'override=True' to replace it.")
        elif event_set.name in self.event_sets and override:
            print(f"Warning: Overriding existing event set with name {event_set.name}")
            
        self.event_sets[event_set.name] = event_set

    def slice_spikes_by_intervals(self, iset_name):
        """Restricts spike data to a specified interval and caches the result.

        This function restricts spike data to a given interval defined by the 
        provided interval name. If the results have been previously computed, 
        they are retrieved from cache to speed up the process.

        Args:
            iset_name (str): The name of the IntervalSet to restrict the spikes to.

        Returns: 
            None. Alters self.interval_to_spikes and self.iset_to_spikes in-place.
        """

        # Fetch the desired interval set (padded if padding has been added to the object).
        interval_set = self.interval_sets[iset_name]
        
        # Return if result is already computed and stored.
        if iset_name in self.iset_to_spikes:
            return
        
        # Restrict the spike train to the interval.
        restricted_spikes = self.spike_train.restrict(interval_set)

        # spike_pointer = 0

        # Initialize a storage for mapped spikes if not present.
        if iset_name not in self.interval_to_spikes: 
            self.interval_to_spikes[iset_name] = {}
        
        # Iterate through the intervals and map spikes.
        for start, end in zip(interval_set.start, interval_set.end):
            # OLD LOGIC
            # interval_spikes = []
            # while spike_pointer < len(restricted_spikes) and restricted_spikes[spike_pointer] <= end:
            #     if restricted_spikes[spike_pointer] >= start:
            #         interval_spikes.append(restricted_spikes[spike_pointer])
            #     spike_pointer += 1
            
            # CHANGED: for spike train as Ts
            interval_spikes = restricted_spikes.get(start=start, end=end)

            self.interval_to_spikes[iset_name][(start, end)] = interval_spikes
        
        # Cache the result for future use.
        self.iset_to_spikes[iset_name] = restricted_spikes


    def compute_log_ISIs(self, iset_name, lock_point=None, scaling_factor=1000):
        """Compute logarithm of inter-spike intervals (ISIs) for a specific interval set and store them.

        This function uses the spike train mapped to a particular interval set
        to compute the logarithm of inter-spike intervals. 
        If an event set name is passed in, the function first time-locks the spike 
        train to the corresponding event that occurs in each interval before computing log ISIs.

        Args:
            iset_name (str): Name of the interval set whose mapped spikes should be used.
            lock_point (str, optional): Either the name of the event set that the spikes should be time-locked to,
                or 'start'/'end' to time-lock to the start/end of each interval.
            scaling_factor (int, optional): Factor to scale the spike times. Defaults to 1000.

        Returns:
            numpy.ndarray: Array containing logISIs for the specified interval.

        Altered Data Structures:
            self.iset_to_log_ISIs: Dictionary storing log ISIs mapped to specific intervals.
        
        Raises:
            KeyError: If the interval_name does not exist in the interval_sets,
            or if lock point is invalid.
        """

        if iset_name not in self.interval_sets:
            raise KeyError("Interval set name doesn't exist")
        
        if lock_point is not None:
            if lock_point not in ['start', 'end'] and lock_point not in self.event_sets:
                raise KeyError("Invalid lock point")

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
            # CHANGED: for spike train as Ts
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


    def time_lock_to_interval(self, iset_name, lock_point):
        """Time-lock spikes to the start or end of intervals.
        
        This function adjusts spike times according to the start or end of the 
        specified intervals, updating the mapped and modified spike trains. It treats "start" and "end" like an event_name for storing. 
        
        Args:
            iset_name (str): Name of the interval set from which unpadded start and end times are retrieved.
            lock_point (str): Either 'start' or 'end', indicating where spikes should be time-locked.
            
        Returns:
            None: Alters self.locked_iset_to_spikes and self.locked_interval_to_spikes in-place.
        """
        
        if lock_point not in ['start', 'end']:
            raise ValueError("'lock_point' should be either 'start' or 'end'")

        # Return if result is already computed and stored.
        if (lock_point, iset_name) in self.locked_iset_to_spikes:
            return

        # OLD LOGIC

        # starts = self.interval_sets[iset_name].start
        # ends = self.interval_sets[iset_name].end
            
        # # Sort the keys of interval_to_spikes for this iset_name based on the start times
        # sorted_keys = sorted(self.interval_to_spikes[iset_name].keys(), key=lambda x: x[0])

        # modified_spikes_agg = []

        # for idx, (start, end) in enumerate(sorted_keys):
            
        #     if lock_point == 'start':
        #         time_adjustment = starts[idx]
        #     else:
        #         time_adjustment = ends[idx]
            
        #     # Adjust spike times based on the lock point
        #     adjusted_spikes = [spike - time_adjustment for spike in self.interval_to_spikes[iset_name][(start, end)]]
            
        #     # Store in the mapped structure
        #     key = (lock_point, iset_name)
        #     if key not in self.modified_spike_trains_mapped:
        #         self.modified_spike_trains_mapped[key] = {}
        #     self.modified_spike_trains_mapped[key][(start, end)] = np.array(adjusted_spikes)
            
        #     modified_spikes_agg.extend(adjusted_spikes)

        # # Convert the accumulated spikes into a Tsd object and store
        # tsd_obj = nap.Tsd(t=np.array(modified_spikes_agg))
        # self.modified_spike_trains[(lock_point, iset_name)] = tsd_obj

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


    def _match_event_to_interval(self, event_set, iset_name):
        """Match each interval of a specific type to a corresponding event in the provided event set.

        Args:
            event_set: Event set to be matched with intervals.
            iset_name (str): Name of the interval type being matched.

        Returns:
            dict: Mapping of intervals to corresponding event timestamps.

        """
        matched_events = {}
        for start, end in self.interval_to_spikes[iset_name].keys():
            events_in_interval = [ts for ts in event_set.events.keys() if start <= ts <= end]
            if len(events_in_interval) != 1:
                raise ValueError(f"Interval ({start}, {end}) doesn't match exactly one event in event_set.")
            matched_events[(start, end)] = events_in_interval[0]

        return matched_events

    def time_lock_to_event(self, eset_name, iset_name):
        """Recenter spike times based on events.

        Args:
            eset_name (str): Name of the Event set to recenter spikes.
            iset_name (str): Name of the interval whose mapped spikes should be used. 

        """

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

        # CHANGED: spike_trains Ts instead of Tsd
        self.locked_iset_to_spikes[(eset_name, iset_name)] = nap.Ts(t=np.array(locked_spikes_agg))


    def save_spikes(self, iset_name, filename, lock_point=None):
        """Save spike train for a specific interval in npz format. Optionally add time-locking.

        Args:
            iset_name (str): Name of the interval set to save a spike train for.
            eset_name (str, optional): Name of the lock_point (an event, or the start/end of the interval)
                to time lock to.
            filename (str): Filename to save the spike train.

        Raises:
            RuntimeError: If the provided filename is invalid.

        Loading Example:
            spike_train = nap.load_file(filename)
        """
        if lock_point is not None:
            if (lock_point, iset_name) not in self.locked_iset_to_spikes:
                raise KeyError("Invalid lock point, or spikes have not been mapped to this interval set and lock point")
        elif iset_name not in self.iset_to_spikes:
            raise KeyError("Interval set doesn't exist, or spikes have not been mapped to this interval set")

        self._validate_filename(filename)
        if lock_point is not None:
            self.locked_iset_to_spikes[(lock_point, iset_name)].save(filename)
        else:
            self.iset_to_spikes[iset_name].save(filename)

        # OLD LOGIC
        # if mapped_filename:
        #     self._validate_filename(mapped_filename)
        #     mapped_spikes_data = self.locked_interval_to_spikes.get((event_name, interval_name), {})
            
        #     if mapped_spikes_data:
        #         intervals, spikes = zip(*mapped_spikes_data.items())
        #         starts, ends = zip(*intervals)
                
        #         np.savez(mapped_filename, 
        #                 event_names=np.array([event_name] * len(starts)), 
        #                 interval_names=np.array([interval_name] * len(starts)), 
        #                 starts=np.asarray(starts), 
        #                 ends=np.asarray(ends), 
        #                 spikes=spikes)
            
        # if spike_train_filename:
        #     spike_train_data = self.locked_interval_to_spikes.get((event_name, interval_name))
        #     if spike_train_data:
        #         spike_train_data.save(spike_train_filename)


    def save_log_ISIs(self, iset_name, filename, lock_point=None):
        """Save computed logISIs for a specific interval to a .npz file. 
        Optionally add time-locking.

        Args:
            iset_name (str): Name of the interval set to save log ISIs for.
            eset_name (str, optional): Name of the lock_point (an event, or the start/end of the interval)
                to time lock to.
            filename (str): Filename to save the log ISIs.


        Raises:
            ValueError: If logISIs haven't been computed for the given event and interval.
            RuntimeError: If the filename is invalid.

        Loading Example:
            with np.load(filename, allow_pickle=True) as data:
                log_ISIs = data['log_ISIs']
        """

        if lock_point is not None:
            if (lock_point, iset_name) not in self.locked_iset_to_log_ISIs:
                raise KeyError("Invalid lock point, or log ISIs have not been computed for this interval and lock point")
        elif iset_name not in self.iset_to_log_ISIs:
            raise KeyError("Interval set doesn't exist, or log ISIs have not been computed for this interval set")

        self._validate_filename(filename)
        if lock_point is not None:
            locked_log_ISIs = self.locked_iset_to_log_ISIs[(lock_point, iset_name)]
            np.savez(filename, locked_log_ISIs=locked_log_ISIs)
        else:
            log_ISIs = self.iset_to_log_ISIs[iset_name]
            np.savez(filename, log_ISIs=log_ISIs)
        
        # OLD LOGIC
        # log_ISIs_data = self.locked_iset_to_log_ISIs.get((event_name, interval_name))

        # if log_ISIs_data is None:
        #     raise ValueError(f"LogISIs not computed for event '{event_name}' and interval '{interval_name}'. Run compute_log_ISIs method first.")
        
        # self._validate_filename(filename)
        # np.savez(filename, log_ISIs=log_ISIs_data)


    @staticmethod
    def _validate_filename(filename):
        """Validate the provided filename.

        Args:
            filename (str): Filename to validate.

        Raises:
            RuntimeError: If the filename is invalid.
        """
        if not isinstance(filename, str):
            raise RuntimeError("Filename should be a string.")
        if os.path.isdir(filename):
            raise RuntimeError(f"{filename} is a directory.")
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            raise RuntimeError(f"Path {directory} doesn't exist.")

