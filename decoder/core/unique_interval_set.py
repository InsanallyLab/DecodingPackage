from multiprocessing.sharedctypes import Value
import pynapple as nap 
import numpy as np  


class UniqueIntervalSet(nap.IntervalSet):
    """
    A subclass of pynapple.IntervalSet ensuring mutable interval sets. 

    Ensures that the intervals added to this set are unique and non-overlapping. 
    Also allows for optional padding of intervals.
    """

    def __init__(self, name, start, end=None, time_units="s", start_padding=None, end_padding=None, **kwargs):
        """
        UniqueIntervalSet initializer.

        Parameters
        ----------
        name : str
            Name of the interval set.
        start : numpy.ndarray or number or pandas.DataFrame or pandas.Series
            Start times of the intervals.
        end : numpy.ndarray or number or pandas.Series, optional if start is a pandas.DataFrame with start and end values
            End times of the intervals.
        time_units : str, optional (default = 's')
            The time units in which intervals are specified. Valid options are 'us', 'ms', 's'.
        start_padding : number, numpy.ndarray, or list, optional
            Padding to add to the start times. Can be a positive or negative value(s).
        end_padding : number, numpy.ndarray, or list, optional
            Padding to add to the end times. Can be a positive or negative value(s).
        **kwargs
            Additional parameters passed to pandas.DataFrame.

        
        If padding is passed in, the padded start/end times are stored as the 
        default self.start, self.end values. Unpadded start/end times are stored
        separately in self.unpadded_start, self.unpadded_end.
        """
        
        super().__init__(start=start, end=end, time_units=time_units, **kwargs)

        self.name = name
        # Note: time_units is not initialized by IntervalSet's init call
        self.time_units = time_units

        # Ensure padding is passed in as a supported data type
        if start_padding is not None and not isinstance(start_padding, (list, np.ndarray, int, float)):
            raise TypeError("Unsupported datatype for start padding passed in.")
        if end_padding is not None and not isinstance(end_padding, (list, np.ndarray, int, float)):
            raise TypeError("Unsupported datatype for end padding passed in.")

        # Check for consistent length of padding if passed in as a list/array
        if isinstance(start_padding, (list, np.ndarray)):
            if len(start_padding) != len(self.start):
                raise ValueError("Length of start_padding inconsistent with length of start intervals.")

        if isinstance(end_padding, (list, np.ndarray)):
            if len(end_padding) != len(self.end):
                raise ValueError("Length of end_padding inconsistent with length of end intervals.")

        # Convert potential single number padding to arrays or lists of consistent length with start and end
        if isinstance(start_padding, (int, float)):
            start_padding = [start_padding] * len(self.start)

        if isinstance(end_padding, (int, float)):
            end_padding = [end_padding] * len(self.end)
        
        self.start_padding = np.asarray(start_padding) if start_padding is not None else None
        self.end_padding = np.asarray(end_padding) if end_padding is not None else None

        self.unpadded_start = self.start if self.start_padding is not None else None
        self.unpadded_end = self.end if self.end_padding is not None else None 

        # Reinitialize IntervalSet object using padded start/end times 
        start = (self.start + self.start_padding) if self.start_padding is not None else self.start
        end = (self.end + self.end_padding) if self.end_padding is not None else self.end
        super().__init__(start=start, end=end)
        
    # def get_padded_interval_set(self):
    #     """Get the padded intervals.

    #     Returns:
    #         UniqueIntervalSet: New interval set with padded intervals.
    #     """
    #     if self.start_padding is not None:
    #         padded_start = self.start - self.start_padding
    #     else:
    #         padded_start = self.start
        
    #     if self.end_padding is not None:
    #         padded_end = self.end + self.end_padding
    #     else:
    #         padded_end = self.end

    #     return UniqueIntervalSet(name=self.name + "_padded", start=padded_start, end=padded_end, time_units=self.time_units)

    def add_interval(self, start, end, start_pad=None, end_pad=None):
        """
        Adds an interval to the unique intervals.

        Parameters:
        start (number): (Unpadded) start time of the new interval.
        end (number): (Unpadded) end time of the new interval.
        start_pad (number): Padding to add to start time if the interval set is padded.
        end_pad (number): Padding to add to the end time if the interval set is padded.
        """
        if start_pad is not None and self.start_padding is None:
            raise ValueError("IntervalSet does not have start padding. Cannot pass in start padding for new interval.")
        if end_pad is not None and self.end_padding is None:
            raise ValueError("IntervalSet does not have end padding. Cannot pass in end padding for new interval.")

        if self.start_padding is not None:
            if start_pad is None:
                raise ValueError("Start padding for new interval is missing")
            else:
                self.start_padding = np.append(self.start_padding, start_pad)
                self.unpadded_start = np.append(self.unpadded_start, start)
        
        if self.end_padding is not None:
            if end_pad is None:
                raise ValueError("End padding for new interval is missing")
            else:
                self.end_padding = np.append(self.end_padding, end_pad)
                self.unpadded_end = np.append(self.unpadded_end, end)
        
        # Append the new interval values to the current set of intervals
        new_start_values = np.append(self.start, start + start_pad) if start_pad is not None else np.append(self.start, start)
        new_end_values = np.append(self.end, end + end_pad) if end_pad is not None else np.append(self.end, end)

        # Reinitialize the IntervalSet with the updated intervals. Need to do so
        # because IntervalSets are immutable.
        super().__init__(start=new_start_values, end=new_end_values)


    def delete_interval(self, start, end):
        """
        Delete a specified interval from the interval set. If the interval set
        is padded, pass in the unpadded start/end times as arguments.

        Args:
            start (number): (Unpadded) start time of the interval to delete.
            end (number): (Unpadded) end time of the interval to delete.

        Raises:
            ValueError: If the specified interval does not exist in the interval sets.
        """

        # Identify the indices of the intervals to keep (i.e., not delete)
        mask = None 
        if self.start_padding is not None:
            if self.end_padding is not None:
                mask = ~((self.unpadded_start == start) & (self.unpadded_end == end)) 
            else:
                mask = ~((self.unpadded_start == start) & (self.end == end)) 
        elif self.end_padding is not None:
            mask = ~((self.start == start) & (self.unpadded_end == end)) 
        else:
            mask = ~((self.start == start) & (self.end == end)) 

        if np.all(mask):
            raise ValueError(f"Specified interval ({start}, {end}) does not exist.")
        
        # Extract the intervals to keep
        new_start_values = self.start[mask]
        new_end_values = self.end[mask]

        # Reinitialize the IntervalSet with the updated intervals. Need to do so
        # because IntervalSets are immutable.
        super().__init__(start=new_start_values, end=new_end_values)

        # Remove deleted interval from padding and padded start/end times, if necessary
        if self.start_padding is not None:
            self.start_padding = self.start_padding[mask]
            assert(self.unpadded_start is not None)
            self.unpadded_start = self.unpadded_start[mask]

        if self.end_padding is not None:
            self.end_padding = self.end_padding[mask]
            assert(self.unpadded_end is not None)
            self.unpadded_end = self.unpadded_end[mask]