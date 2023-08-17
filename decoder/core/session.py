import os
import numpy as np
import pynapple as nap

class Session:

    def __init__(self, spike_times, interval_set=None, event_set=None, name=None, spike_data=None):
        """Initialize the Session class.

        Args:
            spike_times (array-like): Spike times for SpikeTrain.
            interval_set (UniqueIntervalSet, optional): Interval set for the session. Defaults to None.
            event_set (EventSet, optional): Set of events for the session. Defaults to None.
            name (str, optional): Session name. Defaults to None.
            spike_data (array-like, optional): Data associated with the spikes. Defaults to None.
        """
        self.name = name
        self.spike_train = nap.Tsd(spike_times, spike_data)
        self.interval_sets = interval_set
        self.event_set = event_set
        self.mapped_spikes = {}
        self.log_isis = None
        self._cache = {}

    def align_intervals(self, interval_set=None):
        """Align spike times according to intervals and cache the result.

        Args:
            interval_set (object, optional): IntervalSet to align the object. Defaults to self.interval_sets.

        Returns:
            object: Result of the alignment.
        """
        interval_set = interval_set or self.interval_sets
        
        if interval_set in self._cache:
            return self._cache[interval_set]
        
        restricted_spikes = self.spike_train.restrict(interval_set)
        spike_pointer = 0
        for start, end in zip(interval_set.start.values, interval_set.end.values):
            interval_spikes = []
            while spike_pointer < len(restricted_spikes) and restricted_spikes[spike_pointer] <= end:
                if restricted_spikes[spike_pointer] >= start:
                    interval_spikes.append(restricted_spikes[spike_pointer])
                spike_pointer += 1
            self.mapped_spikes[(start, end)] = interval_spikes

        self._cache[interval_set] = restricted_spikes
        return restricted_spikes

    def _match_event_to_interval(self, event_set):
        """Match each interval to a corresponding event in the provided event set.

        Args:
            event_set: Event set to be matched with intervals.

        Returns:
            dict: Mapping of intervals to corresponding event timestamps.

        Raises:
            ValueError: If any interval doesn't match exactly one event.
        """
        matched_events = {}

        for start, end in self.mapped_spikes.keys():
            events_in_interval = [ts for ts in event_set.events.keys() if start <= ts <= end]
            if len(events_in_interval) != 1:
                raise ValueError(f"Interval ({start}, {end}) doesn't match exactly one event in event_set.")
            matched_events[(start, end)] = events_in_interval[0]

        return matched_events

    def recenter_spikes(self, event_set=None):
        """Recenter spike times based on events.

        Args:
            event_set (object, optional): Event set to recenter spikes. Defaults to self.event_set.

        Raises:
            ValueError: If any interval doesn't match exactly one event.
        """
        event_set = event_set or self.event_set
        matched_events = self._match_event_to_interval(event_set)
        all_recentered_spikes = []

        for (start, end), spikes in self.mapped_spikes.items():
            event_time = matched_events[(start, end)]
            recentered_spikes = np.array(spikes) - event_time
            all_recentered_spikes.extend(recentered_spikes.tolist())
            self.mapped_spikes[(start, end)] = recentered_spikes.tolist()

        self.spike_train = nap.Tsd(t=np.array(all_recentered_spikes))

    def compute_log_isis(self, scaling_factor=1000):
        """Compute logarithm of inter-spike intervals (ISIs) and store them.

        Args:
            scaling_factor (int, optional): Factor to scale the spike times. Defaults to 1000.

        Returns:
            numpy.ndarray: Array containing logISIs for each interval.
        """
        log_isis_list = []
        for start, end in zip(self.interval_sets.starts.values, self.interval_sets.ends.values):
            spikes = self.mapped_spikes.get((start, end), [])
            scaled_spikes = [time * scaling_factor for time in spikes]
            interval_isis = np.diff(scaled_spikes)
            interval_log_isis = np.log10(interval_isis) if interval_isis.size != 0 else []
            log_isis_list.append(interval_log_isis)

        self.log_isis = np.array(log_isis_list, dtype='object')
        return self.log_isis

    def save_spike_train(self, mapped_filename=None, spike_train_filename=None):
        """Save spike trains in npz format.

        Args:
            mapped_filename (str, optional): Filename to save the mapped spikes. Defaults to None.
            spike_train_filename (str, optional): Filename to save the SpikeTrain. Defaults to None.

        Raises:
            RuntimeError: If the provided filename is invalid.

        Loading Example:
            If saved using mapped_filename:
                with np.load(mapped_filename) as data:
                    starts = data['starts']
                    ends = data['ends']
                    spikes = data['spikes']

            If saved using spike_train_filename:
                spike_train = nap.Tsd.load(spike_train_filename)
        """
        if mapped_filename:
            self._validate_filename(mapped_filename)
            intervals, spikes = zip(*self.mapped_spikes.items())
            starts, ends = zip(*intervals)
            np.savez(mapped_filename, starts=np.array(starts), ends=np.array(ends), spikes=spikes)
        
        if spike_train_filename:
            self.spike_train.save(spike_train_filename)

    def save_log_isis(self, filename="log_isis.npz"):
        """Save computed logISIs to a .npz file.

        Args:
            filename (str, optional): Output filename. Defaults to "log_isis.npz".

        Raises:
            ValueError: If logISIs haven't been computed.
            RuntimeError: If the filename is invalid.

        Loading Example:
            with np.load(filename) as data:
                log_isis = data['log_isis']
        """
        if self.log_isis is None:
            raise ValueError("LogISIs not computed. Run compute_log_isis method first.")
        self._validate_filename(filename)
        np.savez(filename, log_isis=self.log_isis)


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

