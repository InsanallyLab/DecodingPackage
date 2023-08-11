import pynapple as nap
import numpy as np

class Session:
    def __init__(self, name, spike_time, start, spike_data=None, end=None):
        """
        Initialize the Session class with SpikeTrain, IntervalSets, and an empty list for EventSets.
        
        Parameters:
        name (str): Name of the session
        spike_time (array-like): Times of spikes for SpikeTrain
        start (array-like): Start times of intervals for IntervalSets
        spike_data (array-like, optional): Data associated with the spikes for SpikeTrain
        end (array-like, optional): End times of intervals for IntervalSets
        """
        self.name = name
        self.SpikeTrain = nap.Ts(spike_time) if spike_data is None else nap.Tsd(spike_time, spike_data)
        self.IntervalSets = nap.IntervalSet(start, end)  # Using PyNapple's IntervalSet
        self.EventSets = []  # List to hold custom event sets
        self._cache = {}  # Dictionary to hold cached data about aligned intervals

    def add_interval(self, start, end):
        """
        Adds an interval to the IntervalSets ensuring it obeys the conditions specified in the IntervalSet class.
        
        Parameters:
        start (float): Start time of the new interval
        end (float): End time of the new interval

        Raises:
        ValueError: If start time is less than previous end time or if end time is less than or equal to start time
        """
        if len(self.IntervalSets) > 0:
            last_end = self.IntervalSets.ends.values[-1]  # Accessing underlying values
            if start < last_end:
                raise ValueError("Start time must be greater than the previous end time")
        if end <= start:
            raise ValueError("End time must be greater than the start time")

        new_starts = np.concatenate([self.IntervalSets.starts.values, [start]])  # Accessing underlying values
        new_ends = np.concatenate([self.IntervalSets.ends.values, [end]])  # Accessing underlying values
        self.IntervalSets = nap.IntervalSet(start=new_starts, end=new_ends, time_units=self.IntervalSets.time_units)

    def add_event_set(self, event_set):
        """
        Add a custom event set to the Session (e.g., a lick is an event).

        Parameters:
        event_set (object): The custom event set to be added
        """
        self.EventSets.append(event_set)

    def interval_alignment(self, interval_set=None):
        """
        Align intervals and cache the results. If already cached, return the cached result.

        Parameters:
        interval_set (object, optional): The IntervalSet to be aligned. Defaults to self.IntervalSets

        Returns:
        object: Result of the alignment
        """
        if interval_set is None:
            interval_set = self.IntervalSets
        if interval_set in self._cache:
            return self._cache[interval_set]
        else:
            result = self.SpikeTrain.restrict(interval_set)
            self._cache[interval_set] = result
            return result

    def transform(self, spikes, trials):
        """
        Take spikes and trials and convert to logISIs.
        TODO: change to take in IntervalSets and return isis for multiple trials

        Parameters:
        spikes (array-like): The spikes data
        trials (array-like): The trials data
        """
        pass
