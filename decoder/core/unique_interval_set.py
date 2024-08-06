import pynapple as nap 
import numpy as np  
from typing import Optional, Union
from numpy.typing import ArrayLike

class UniqueIntervalSet(nap.IntervalSet):
    """
    A subclass of pynapple.IntervalSet ensuring mutable interval sets. Allows
    for optional padding of intervals.
    Ensures that the intervals added to this set are unique and non-overlapping. 
    """

    def __init__(
        self, 
        name: str, 
        start, 
        end, 
        time_units: str = "s", 
        fill_end_nans : Optional[str] = None,
        remove_overlaps: Optional[str] = None,
        start_padding: Optional[Union[int, float, ArrayLike]] = None, 
        end_padding: Optional[Union[int, float, ArrayLike]] = None, 
        **kwargs):
        """
        Parameters
        ----------
        name : str
            Name of the interval set.
        start : number or array-like
            Start times of the intervals.
        end : number or array-like
            End times of the intervals. 
        time_units : str, optional (default = 's')
            The time units in which intervals are specified. 
            Valid options are 'us', 'ms', 's'.
        fill_end_nans : str, optional
            Specifies how to fill NaN values in the end times. 
            "mean": calculates the mean difference between start and end times 
                in the interval set, and adds this mean to the start times to 
                fill NaN values. 
            "min": calculates the min difference between start and end times 
                in the interval set, and adds this min to the start times to 
                fill NaN values. 
            "max": calculates the max difference between start and end times 
                in the interval set, and adds this max to the start times to 
                fill NaN values. 
        remove_overlaps : str, optional
            If two intervals overlap, specifies which of the two to remove. 
            Pynapple will merge the two intervals by default. 
            'first': removes the earlier of the two intervals.
            'last': removes the later of the two intervals.
        start_padding : number, numpy.ndarray, or list, optional
            Padding to add to the start times. Can be a positive or negative value(s).
        end_padding : number, numpy.ndarray, or list, optional
            Padding to add to the end times. Can be a positive or negative value(s).
        **kwargs
            Additional parameters passed to IntervalSet initialization.

        If padding is passed in, the unpadded start/end times are stored as the 
        default self.start, self.end values. Padded start/end times are stored
        separately in self.padded_start, self.padded_end.
        """
        if np.any(np.isnan(start)):
            raise ValueError("Interval start times should not include NaN.")
        if np.any(np.isnan(end)) and fill_end_nans is None:
            raise ValueError("Interval end times should not include NaN. Set fill_end_nans accordingly to fill in NaN values.")

        if fill_end_nans is not None:
            if fill_end_nans == "mean":
                start_end_diffs = end - start
                avg_start_end_diff = np.nanmean(start_end_diffs)
                end[np.isnan(end)] = start[np.isnan(end)] + avg_start_end_diff
            elif fill_end_nans == "min":
                start_end_diffs = end - start
                min_start_end_diff = np.nanmin(start_end_diffs)
                end[np.isnan(end)] = start[np.isnan(end)] + min_start_end_diff
            elif fill_end_nans == "max":
                start_end_diffs = end - start
                max_start_end_diff = np.nanmax(start_end_diffs)
                end[np.isnan(end)] = start[np.isnan(end)] + max_start_end_diff
            else: 
                raise KeyError("Invalid argument for fill_end_nans. The only valid arguments are 'mean', 'min', and 'max'.")
        
        self.idx_removed = None
        idx_to_remove = []
        for i in range(1, len(start)):
            if start[i] < end[i - 1]:
                if remove_overlaps is None:
                    print("Overlapping intervals are present. Pynapple will automatically merge these intervals together. Pass in an argument for remove_overlaps to remove one of the intervals instead of merging them.")
                elif remove_overlaps == "first":
                    idx_to_remove.append(i - 1)
                elif remove_overlaps == "last":
                    idx_to_remove.append(i)
                else:
                    raise KeyError("Invalid argument for remove_overlaps")
        if len(idx_to_remove) > 0:
            start = np.delete(start, idx_to_remove)
            end = np.delete(end, idx_to_remove)
            self.idx_removed = idx_to_remove

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

        '''
            Pynapple's IntervalSet doesn't allow for overlap, but padded 
            portions should be allowed to overlap with each other. 
        '''
        self.padded_start = (self.start + self.start_padding) if self.start_padding is not None else None
        self.padded_end = (self.end + self.end_padding) if self.end_padding is not None else None 

    def add_interval(
        self, 
        start: Union[int, float], 
        end: Union[int, float], 
        start_pad: Optional[Union[int, float]] = None, 
        end_pad: Optional[Union[int, float]] = None):
        """
        Adds an interval to the unique interval set.

        Parameters
        ----------
        start : number
            (Unpadded) start time of the new interval.
        end : number 
            (Unpadded) end time of the new interval.
        start_pad : number
            Padding to add to start time if the interval set is padded.
        end_pad : number 
            Padding to add to the end time if the interval set is padded.
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
                self.padded_start = np.append(self.padded_start, start + start_pad)
        
        if self.end_padding is not None:
            if end_pad is None:
                raise ValueError("End padding for new interval is missing")
            else:
                self.end_padding = np.append(self.end_padding, end_pad)
                self.padded_end = np.append(self.padded_end, end + end_pad)
        
        # Append the new interval values to the current set of intervals
        new_start_values = np.append(self.start, start)
        new_end_values = np.append(self.end, end) 

        # Reinitialize the IntervalSet with the updated intervals. Need to do so
        # because IntervalSets are immutable.
        super().__init__(start=new_start_values, end=new_end_values)


    def delete_interval(
        self, 
        start: Union[int, float],
        end: Union[int, float]):
        """
        Deletes a specified interval from the interval set. If the interval set
        is padded, pass in the unpadded start/end times as arguments.

        Parameters
        ----------
        start : number
            (Unpadded) start time of the interval to delete.
        end : number
            (Unpadded) end time of the interval to delete.
        """

        # Identify the indices of the intervals to keep (i.e not delete)
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
            assert(self.padded_start is not None)
            self.padded_start = self.padded_start[mask]

        if self.end_padding is not None:
            self.end_padding = self.end_padding[mask]
            assert(self.padded_end is not None)
            self.padded_end = self.padded_end[mask]