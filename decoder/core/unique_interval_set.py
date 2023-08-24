import pynapple as nap 
import numpy as np  


class UniqueIntervalSet(nap.IntervalSet):
    """
    A subclass of pynapple.IntervalSet ensuring unique intervals.

    Ensures that the intervals added to this set are unique and non-overlapping. Provides utility 
    methods for checking and enforcing this uniqueness. Also allows for optional padding of intervals.
    """

    def __init__(self, name, start, end=None, time_units="s", start_padding=None, end_padding=None, **kwargs):
        """
        UniqueIntervalSet initializer.

        Parameters
        ----------
        name : str
            Name of the interval set.
        start : numpy.ndarray, number, or pandas.DataFrame
            Start times of the intervals.
        end : numpy.ndarray or number, optional
            End times of the intervals.
        time_units : str, optional (default = 's')
            The time units in which intervals are specified. Valid options are 'us', 'ms', 's'.
        start_padding : number, numpy.ndarray, or list, optional
            Padding to subtract from the start times. Must be a positive value(s).
        end_padding : number, numpy.ndarray, or list, optional
            Padding to add to the end times. Must be a positive value(s).
        **kwargs
            Additional parameters passed to pandas.DataFrame.
        """
        self.name = name
        
        # Convert potential single number padding to arrays or lists of consistent length with start and end
        if isinstance(start_padding, (int, float)):
            start_padding = [start_padding] * len(start)
        if isinstance(end_padding, (int, float)):
            end_padding = [end_padding] * len(end)
        
        self.start_padding = np.array(start_padding) if start_padding is not None else None
        self.end_padding = np.array(end_padding) if end_padding is not None else None
        
        # Check for consistent length of start, end, and paddings
        if self.start_padding is not None and len(self.start_padding) != len(start):
            raise ValueError("Length of start_padding must be consistent with length of start intervals.")
        if self.end_padding is not None and len(self.end_padding) != len(end):
            raise ValueError("Length of end_padding must be consistent with length of end intervals.")
            
        super().__init__(start=start, end=end, time_units=time_units, **kwargs)

    def get_padded_interval(self):
        """Get the padded intervals.

        Returns:
            UniqueIntervalSet: New interval set with padded intervals.
        """
        if self.start_padding is not None:
            padded_start = self.starts.values - self.start_padding
        else:
            padded_start = self.starts.values
        
        if self.end_padding is not None:
            padded_end = self.ends.values + self.end_padding
        else:
            padded_end = self.ends.values

        return UniqueIntervalSet(name=self.name + "_padded", start=padded_start, end=padded_end, time_units=self.time_units)


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