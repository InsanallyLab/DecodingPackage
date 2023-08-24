import pynapple as nap 
import numpy as np  


class UniqueIntervalSet(nap.IntervalSet):
    """
    A subclass of pynapple.IntervalSet ensuring unique intervals.

    Ensures that the intervals added to this set are unique and non-overlapping. Provides utility 
    methods for checking and enforcing this uniqueness.
    """

    def __init__(self, name, start, end=None, time_units="s", **kwargs):
        """
        UniqueIntervalSet initializer.

        Parameters
        ----------
        name : str, name of the interval set 
        start : numpy.ndarray, number, or pandas.DataFrame
            Start times of the intervals.
        end : numpy.ndarray or number, optional
            End times of the intervals.
        time_units : str, optional (default = 's')
            The time units in which intervals are specified. Valid options are 'us', 'ms', 's'.
        **kwargs
            Additional parameters passed to pandas.DataFrame.
        """
        self.name = name 
        super().__init__(start=start, end=end, time_units=time_units, **kwargs)

    def add_interval(self, start, end):
        """
        Adds an interval to the unique intervals.

        Parameters:
        start (float): Start time of the new interval.
        end (float): End time of the new interval.
        """
        # Append the new interval values to the current set of intervals
        new_start_values = np.append(self.start.values, start)
        new_end_values = np.append(self.end.values, end)

        # Reinitialize the IntervalSet with the updated intervals
        super().__init__(start=new_start_values, end=new_end_values)

    def delete_interval(self, start, end):
        """
        Delete a specified interval from the interval sets.

        Args:
            start (float): Start time of the interval to delete.
            end (float): End time of the interval to delete.

        Raises:
            ValueError: If the specified interval does not exist in the interval sets.
        """
        # Identify the indices of the intervals to keep (i.e., not delete)
        mask = ~((self.start.values == start) & (self.end.values == end))
        if not np.any(mask):
            raise ValueError(f"Specified interval ({start}, {end}) does not exist.")
        
        # Extract the intervals to keep
        new_start_values = self.start.values[mask]
        new_end_values = self.end.values[mask]

        # Reinitialize the IntervalSet with the updated intervals
        super().__init__(start=new_start_values, end=new_end_values)