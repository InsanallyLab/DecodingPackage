import numpy as np
import pandas as pd
from pynapple import IntervalSet

class UniqueIntervalSet(IntervalSet):
    """
    A subclass of pynapple.IntervalSet ensuring unique intervals.

    Ensures that the intervals added to this set are unique and non-overlapping. Provides utility 
    methods for checking and enforcing this uniqueness.
    """

    def __init__(self, start, end=None, time_units="s", **kwargs):
        """
        UniqueIntervalSet initializer.

        Parameters
        ----------
        start : numpy.ndarray, number, or pandas.DataFrame
            Start times of the intervals.
        end : numpy.ndarray or number, optional
            End times of the intervals.
        time_units : str, optional (default = 's')
            The time units in which intervals are specified. Valid options are 'us', 'ms', 's'.
        **kwargs
            Additional parameters passed to pandas.DataFrame.
        """
        super().__init__(start=start, end=end, time_units=time_units, **kwargs)
        if not (start == [] or end == []): 
            self.ensure_unique_intervals()

    def ensure_unique_intervals(self):
        """
        Ensures that the intervals in the set are unique.

        This method sorts the intervals based on their start times, then checks if any interval 
        overlaps with the next. If overlaps are found, only the first interval in each set of 
        overlapping intervals is kept.
        """
        # Sort intervals based on start times
        self.sort_values(by='start', inplace=True)

        # Drop intervals that overlap with the next
        mask = np.append(True, self['start'].values[1:] >= self['end'].values[:-1])
        self.drop(self.index[~mask], inplace=True)

        # Resetting the index for consistency
        self.reset_index(drop=True, inplace=True)