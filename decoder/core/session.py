import pynapple as nap
import numpy as np
from unique_interval_set import UniqueIntervalSet

class Session:
    def __init__(self, name, spike_time, start, spike_data=None, end=None):
        """
        Initialize the Session class with SpikeTrain, UniqueIntervalSets, and an empty list for EventSets.
        
        Parameters:
        name (str): Name of the session
        spike_time (array-like): Times of spikes for SpikeTrain
        start (array-like): Start times of intervals for UniqueIntervalSets
        spike_data (array-like, optional): Data associated with the spikes for SpikeTrain
        end (array-like, optional): End times of intervals for UniqueIntervalSets
        """
        self.name = name
        self.SpikeTrain = nap.Ts(spike_time) if spike_data is None else nap.Tsd(spike_time, spike_data)
        self.IntervalSets = UniqueIntervalSet(start=start, end=end)
        self._cache = {}  # Dictionary to hold cached data about aligned intervals

    def add_interval(self, start, end):
        """
        Adds an interval to the UniqueIntervalSets.
        
        Parameters:
        start (float): Start time of the new interval
        end (float): End time of the new interval
        """
        # Create a temporary UniqueIntervalSet to combine with existing intervals
        new_interval_set = UniqueIntervalSet(start=[start], end=[end])
        combined_intervals = UniqueIntervalSet(
            start=np.concatenate([self.IntervalSets["start"].values, new_interval_set["start"].values]),
            end=np.concatenate([self.IntervalSets["end"].values, new_interval_set["end"].values])
        )
        
        self.IntervalSets = combined_intervals

    def delete_interval(self, start, end):
        """
        Delete a specified interval from the IntervalSets.

        Args:
            start (float): Start time of the interval to delete.
            end (float): End time of the interval to delete.

        Raises:
            ValueError: If the specified interval does not exist in the IntervalSets.
        """
        # Convert the intervals to lists for easy manipulation
        starts_list = list(self.IntervalSets.starts.values)
        ends_list = list(self.IntervalSets.ends.values)

        # Check if the specified interval exists in the IntervalSets
        if start in starts_list and end in ends_list:
            index = starts_list.index(start)  # Get the index of the interval
            del starts_list[index]  # Delete start time
            del ends_list[index]  # Delete end time
        else:
            raise ValueError(f"Specified interval ({start}, {end}) does not exist.")

        # Update the IntervalSets with the modified lists
        self.IntervalSets = nap.IntervalSet(start=np.array(starts_list), end=np.array(ends_list), time_units=self.IntervalSets.time_units)


    def interval_alignment(self, interval_set=None):
        """
        Align spike times according to a set of intervals. If already cached, return the cached result.

        Parameters:
        interval_set (object, optional): The IntervalSet used to align the object. Defaults to self.IntervalSets

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

    def _ensure_single_event_per_interval(self, event_set):
        """Ensure that each interval contains exactly one event from the provided event set.

        Args:
            event_set (EventSet): The event set containing the events to be checked.

        Raises:
            ValueError: If an interval does not contain exactly one event.
        """
        for start, end in zip(self.IntervalSets.starts, self.IntervalSets.ends):
            # Counting number of events within the interval
            events_in_interval = [time for time in event_set.get_events() if start <= time <= end]
            
            if len(events_in_interval) != 1:
                raise ValueError(f"Interval ({start}, {end}) does not contain exactly one event.")

    def recenter_spikes(self, event_set):
        """Recenter spike times based on the timing of the events in the provided event set.

        Args:
            event_set (EventSet): The event set whose events will be used for recentering.

        Returns:
            nap.Ts: Recentered spike times.
        """
        self._ensure_single_event_per_interval(event_set)

        recentered_spikes = []

        for start, end in zip(self.IntervalSets.starts, self.IntervalSets.ends):
            # Getting the event time within the current interval
            event_time = next(time for time in event_set.get_events() if start <= time <= end)

            # Restricting spikes to the current interval
            spikes_in_interval = self.SpikeTrain.time_slice(start, end).times

            # Recentering spikes based on the event time
            recentered_spikes_interval = spikes_in_interval - event_time

            recentered_spikes.extend(recentered_spikes_interval)

        return nap.Ts(recentered_spikes)  # Returning recentered spike times

    def transform(self, spikes, trials):
        """
        Take spikes and trials and convert to logISIs.
        TODO: change to take in UniqueIntervalSets and return isis for multiple trials

        Parameters:
        spikes (array-like): The spikes data
        trials (array-like): The trials data
        """
        pass
