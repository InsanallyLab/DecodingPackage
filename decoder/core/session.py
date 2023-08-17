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
        self.mapped_spike_train = {}
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
            Align spike times according to a set of intervals and cache the result.

            Parameters:
            interval_set (object, optional): The IntervalSet used to align the object. Defaults to self.IntervalSets.

            Returns:
            object: Result of the alignment.
            """
            if interval_set is None:
                interval_set = self.IntervalSets
            
            if interval_set in self._cache:
                return self._cache[interval_set]
            else:
                restricted_spikes = self.SpikeTrain.restrict(interval_set)

                # Optimized dictionary creation to map intervals to spike times
                spike_pointer = 0
                for start, end in zip(interval_set.start.values, interval_set.end.values):
                    spikes_in_interval = []

                    while spike_pointer < len(restricted_spikes) and restricted_spikes[spike_pointer] <= end:
                        if restricted_spikes[spike_pointer] >= start:
                            spikes_in_interval.append(restricted_spikes[spike_pointer])
                        spike_pointer += 1

                    self.mapped_spike_train[(start, end)] = spikes_in_interval

                self._cache[interval_set] = restricted_spikes
                return restricted_spikes

    def _ensure_single_event_per_interval(self, event_set):
        """Ensure there's exactly one event per interval in the provided event set.

        Args:
            event_set: The event set with events to be matched with intervals.

        Returns:
            dict: Mapping of each interval to its corresponding event timestamp.

        Raises:
            ValueError: If any interval doesn't have exactly one corresponding event.
        """
        interval_event_map = {}

        for start, end in self.mapped_spike_train.keys():
            matching_events = [timestamp for timestamp in event_set.events.keys() if start <= timestamp <= end]
            
            if len(matching_events) != 1:
                raise ValueError(f"Interval ({start}, {end}) does not have exactly one event in the event_set.")
            
            interval_event_map[(start, end)] = matching_events[0]

        return interval_event_map

    def recenter_spikes(self, event_set):
        """Recenter spike times based on events and update internal structures.

        Args:
            event_set: The event set with events to recenter the spikes around.

        Raises:
            ValueError: If any interval doesn't have exactly one corresponding event.
        """
        # Ensure that each interval has one and only one event and get mapping
        interval_event_map = self._ensure_single_event_per_interval(event_set)

        all_recentered_spikes = []

        # Recenter spikes around the event for each interval
        for (start, end), spikes in self.mapped_spike_train.items():
            # Fetching the event time for the interval from the map
            event_time = interval_event_map[(start, end)]

            # Recentering using numpy for efficiency
            recentered_spikes_for_interval = np.array(spikes) - event_time
            
            all_recentered_spikes.extend(recentered_spikes_for_interval.tolist())
            self.mapped_spike_train[(start, end)] = recentered_spikes_for_interval.tolist()

        # Update the SpikeTrain with recentered spike times
        self.SpikeTrain = nap.Tsd(t=np.array(all_recentered_spikes))

    def transform(self, spikes, trials):
        """
        Take spikes and trials and convert to logISIs.
        TODO: change to take in UniqueIntervalSets and return isis for multiple trials

        Parameters:
        spikes (array-like): The spikes data
        trials (array-like): The trials data
        """
        pass
