import pynapple as nap
import numpy as np
import os 
from unique_interval_set import UniqueIntervalSet

class Session:
    def __init__(self,spike_time, start,name = None, spike_data=None, end=None):
        """
        Initialize the Session class with SpikeTrain, UniqueIntervalSets, and an empty list for EventSets.
        
        Parameters:
        spike_time (array-like): Times of spikes for SpikeTrain
        start (array-like): Start times of intervals for UniqueIntervalSets
        name (str, optional): Name of the session
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
        """Recenter spike times based on events and update internal structures. TODO: Assumes that interval_alignment function has been called for the creation of mapped_spike_train

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

    def transform(self, scaling_factor=1000, log_base=10):
        """
        Computes the logarithm of the inter-spike intervals (ISIs) for given time intervals and
        stores them in self.log_isis. TODO: also assumes interval_alignment has been called

        Args:
            scaling_factor (int, optional): Factor by which to scale the spike times. Default is 1000. (in old code it was 1000/ sample_frequency)
            log_base (int, optional): The base of the logarithm to use. Default is 10.

        Returns:
            numpy.ndarray: An array containing logISIs for each interval in IntervalSets.

        """ 
        
        logISIs = []

        # Loop over intervals using the start and end values
        for start_time, end_time in zip(self.IntervalSets.starts.values, self.IntervalSets.ends.values):
            # Extract spike times within the interval using mapped_spike_train
            spiketimes = self.mapped_spike_train.get((start_time, end_time), [])
            
            spiketimes_scaled = [time * scaling_factor for time in spiketimes]
            ISIs_for_interval = np.diff(spiketimes_scaled)
            logISIs_for_interval = np.log10(ISIs_for_interval) if ISIs_for_interval.size != 0 else []
            logISIs.append(logISIs_for_interval)

        self.log_isis = np.array(logISIs, dtype='object')
        
        return self.log_isis

    def save_spike_train(self, mapped_spike_train_filename=None, spike_train_filename=None):
        """
        Save mapped_spike_train dictionary and SpikeTrain (Tsd object) in npz format if their 
        respective filenames are provided.

        Parameters
        ----------
        mapped_spike_train_filename : str, optional
            The filename to store the mapped_spike_train data.
            If None (default), mapped_spike_train will not be saved.
            
        spike_train_filename : str, optional
            The filename to store the SpikeTrain (Tsd object) data.
            If None (default), SpikeTrain will not be saved.

        Raises
        ------
        RuntimeError
            If any filename is not str, path does not exist, or filename is a directory.
        
        Examples
        --------
        >>> session = YourClass()
        >>> session.save_spike_train("path/to/mapped_spike_train.npz", "path/to/spike_train.npz")

        To load the saved mapped_spike_train:

        >>> loaded_data = np.load("path/to/mapped_spike_train.npz")
        >>> starts = loaded_data['starts']
        >>> ends = loaded_data['ends']
        >>> spikes = loaded_data['spikes']

        To load the saved SpikeTrain (Tsd object):

        >>> spike_train_data = np.load("path/to/spike_train.npz")
        >>> time_support = nap.IntervalSet(spike_train_data['start'], spike_train_data['end'])
        >>> spike_train = nap.Tsd(t=spike_train_data['t'], d=spike_train_data['d'], time_support=time_support)
        """
        if mapped_spike_train_filename:
            if not isinstance(mapped_spike_train_filename, str):
                raise RuntimeError("Invalid type; please provide mapped_spike_train_filename as string")

            if os.path.isdir(mapped_spike_train_filename):
                raise RuntimeError("Invalid filename input. {} is directory.".format(mapped_spike_train_filename))

            dirname = os.path.dirname(mapped_spike_train_filename)

            if len(dirname) and not os.path.exists(dirname):
                raise RuntimeError("Path {} does not exist.".format(os.path.dirname(mapped_spike_train_filename)))
            
            # Save mapped_spike_train dictionary
            intervals = list(self.mapped_spike_train.keys())
            spikes = list(self.mapped_spike_train.values())
            
            starts, ends = zip(*intervals)
            
            np.savez(
                mapped_spike_train_filename,
                starts=np.array(starts),
                ends=np.array(ends),
                spikes=spikes
            )
        
        if spike_train_filename:
            # Save SpikeTrain (Tsd object)
            self.SpikeTrain.save(spike_train_filename)

    def save_logISIs(self, filename="log_isis.npz"):
        """
        Saves the computed logISIs to a .npz file.

        Parameters
        ----------
        filename : str, optional
            Path to the output .npz file. Defaults to "log_isis.npz".

        Raises
        ------
        ValueError:
            If `self.log_isis` hasn't been computed yet.

        RuntimeError:
            If filename is not str, path does not exist or filename is a directory.
        
        Examples
        --------
        >>> session = YourClass()
        >>> session.interval_alignment()
        >>> session.transform()
        >>> session.save_logISIs("output_log_isis.npz")

        To load the saved data:

        >>> loaded_data = np.load("output_log_isis.npz", allow_pickle=True)
        >>> log_isis_array = loaded_data['log_isis']
        """

        if self.log_isis is None:
            raise ValueError("logISIs have not been computed yet. Please run the transform method first.")

        if not isinstance(filename, str):
            raise RuntimeError("Invalid type; please provide filename as string")

        if os.path.isdir(filename):
            raise RuntimeError("Invalid filename input. {} is directory.".format(filename))

        dirname = os.path.dirname(filename)

        if len(dirname) and not os.path.exists(dirname):
            raise RuntimeError("Path {} does not exist.".format(os.path.dirname(filename)))

        np.savez(filename, log_isis=self.log_isis)

