# tests/core/test_unique_interval_set.py

from tracemalloc import start
import pytest
from decoder.core.unique_interval_set import UniqueIntervalSet
import pandas as pd
import numpy as np 

'''
Unpadded interval unit tests.
'''
### Initialization tests ###
def test_init_without_overlaps():
    """Tests if UniqueIntervalSet initializes correctly without overlapping intervals."""
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]
    
    interval_set = UniqueIntervalSet(
        name="standard_test", 
        start=start_times, 
        end=end_times)
    
    assert np.array_equal(interval_set.start, start_times)
    assert np.array_equal(interval_set.end, end_times)

def test_init_with_dataframe():
    """Tests if UniqueIntervalSet initializes correctly with a Pandas DataFrame as input."""
    start_end_times = [[0, 5], [10, 15], [20, 25]]
    dataframe = pd.DataFrame(start_end_times, columns=['start', 'end'])

    interval_set = UniqueIntervalSet(name="dataframe_test", start=dataframe)
    
    assert np.array_equal(interval_set.start, [0, 10, 20])
    assert np.array_equal(interval_set.end, [5, 15, 25])

def test_init_with_unnamed_dataframe():
    """Tests if UniqueIntervalSet raises an error if a Pandas DataFrame with an 
    incorrect format (missing column names) is passed in."""
    start_end_times = [[0, 5], [10, 15], [20, 25]]
    dataframe = pd.DataFrame(start_end_times)

    # Missing column names ("start", "end")
    with pytest.raises(AssertionError):
        interval_set = UniqueIntervalSet(
            name="unnamed_dataframe_test", 
            start=dataframe)

def test_init_with_incomplete_dataframe():
    """Tests if UniqueIntervalSet raises an error if a Pandas DataFrame with
    missing end times is passed in."""
    start_times = [0, 10, 20]
    dataframe = pd.DataFrame(start_times, columns=['start'])

    with pytest.raises(AssertionError):
        interval_set = UniqueIntervalSet(
            name="incomplete_dataframe_test", 
            start=dataframe)

def test_init_with_panda_series():
    """Tests if UniqueIntervalSet initializes correctly with a Pandas Series as input."""
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]

    start_series = pd.Series(start_times)
    end_series = pd.Series(end_times)
    
    interval_set = UniqueIntervalSet(
        name="series_test", 
        start=start_series, 
        end=end_series)
    
    assert np.array_equal(interval_set.start, start_times)
    assert np.array_equal(interval_set.end, end_times)

def test_init_missing_end_series():
    start_times = [0, 10, 20]

    start_series = pd.Series(start_times)

    with pytest.raises(AssertionError):
        interval_set = UniqueIntervalSet(
            name="missing_end_test", 
            start=start_series)
    
def test_init_with_ndarray():
    start_times = np.array([0, 10, 20])
    end_times = np.array([5, 15, 25])
    
    interval_set = UniqueIntervalSet(
        name="standard_test", 
        start=start_times, 
        end=end_times)
    
    assert np.array_equal(interval_set.start, start_times)
    assert np.array_equal(interval_set.end, end_times)

def test_init_with_overlaps():
    # Given some overlapping intervals
    start_times = [0, 5, 10, 15, 20]
    end_times = [12, 17, 22, 27, 30]

    # When we create a UniqueIntervalSet with them
    interval_set = UniqueIntervalSet(name="overlap_test", start=start_times, end=end_times)

    # Then it should merge overlapping intervals
    # Here, all intervals overlap, so they should be merged into one
    # Note: Pynapple may output warning that "some starts precede the previous 
    # end", and so it is joining these intervals
    assert interval_set.start[0] == 0
    assert interval_set.end[0] == 30

def test_init_with_non_unique_intervals():
    # Given some non-unique intervals
    start_times = [0, 15, 0]
    end_times = [12, 20, 12]

    # When we create a UniqueIntervalSet with them
    interval_set = UniqueIntervalSet(name="non_unique_test", start=start_times, end=end_times)

    # Then it should merge non-unique intervals
    assert np.array_equal(interval_set.start, [0, 15])
    assert np.array_equal(interval_set.end, [12, 20])

def test_empty_init():
    """Test if UniqueIntervalSet initializes correctly with empty start and end lists."""
    interval_set = UniqueIntervalSet(name="test_empty", start=[], end=[])    
    assert len(interval_set.start) == len(interval_set.end) == 0

def test_mismatched_start_end_lengths():
    """Test if AssertionError is raised when start and end lists are of different lengths."""
    with pytest.raises(AssertionError):
        interval_set = UniqueIntervalSet(
            name="test_mismatched_lens", 
            start=[0, 10], 
            end=[5])

def test_init_negative_times():
    start_times = [-3, -1, 4]
    end_times = [-2, 2, 5]
    
    interval_set = UniqueIntervalSet(
        name="negative_times_test", 
        start=start_times, 
        end=end_times)
    
    assert np.array_equal(interval_set.start, start_times)
    assert np.array_equal(interval_set.end, end_times)

def test_illegal_start_end_times():
    start_times = [2, 5, 20]
    end_times = [1, 4, 25]
    
    # IntervalSet will drop intervals with start times preceding end times
    interval_set = UniqueIntervalSet(
        name="illegal_times_test", 
        start=start_times, 
        end=end_times)

    assert interval_set.start[0] == 20
    assert interval_set.end[0] == 25

### Add interval tests ###
def test_add_non_overlapping_interval():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]
    
    interval_set = UniqueIntervalSet(
        name="add_non_overlapping_test", 
        start=start_times, 
        end=end_times)

    interval_set.add_interval(30, 32)
    assert np.array_equal(interval_set.start, [0, 10, 20, 30])
    assert np.array_equal(interval_set.end, [5, 15, 25, 32])

def test_add_overlapping_interval():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]
    
    interval_set = UniqueIntervalSet(
        name="add_overlapping_test", 
        start=start_times, 
        end=end_times)

    interval_set.add_interval(2.4, 4.8)
    assert np.array_equal(interval_set.start, start_times)
    assert np.array_equal(interval_set.end, end_times)

def test_add_non_unique_interval():
    start_times = [0.4, 10.6, 20.9]
    end_times = [5.3, 15.8, 25.3]
    
    interval_set = UniqueIntervalSet(
        name="add_non_unique_test", 
        start=start_times, 
        end=end_times)

    interval_set.add_interval(10.6, 15.8)
    assert np.array_equal(interval_set.start, start_times)
    assert np.array_equal(interval_set.end, end_times)


### Delete interval tests ###
def test_delete_existing_interval():
    start_times = [0.4, 10.6, 20.9]
    end_times = [5.3, 15.8, 25.3]
    
    interval_set = UniqueIntervalSet(
        name="delete_existing_test", 
        start=start_times, 
        end=end_times)

    interval_set.delete_interval(10.6, 15.8)
    assert np.array_equal(interval_set.start, [0.4, 20.9])
    assert np.array_equal(interval_set.end, [5.3, 25.3])

def test_delete_non_existing_interval():
    start_times = [0.4, 10.6, 20.9]
    end_times = [5.3, 15.8, 25.3]
    
    interval_set = UniqueIntervalSet(
        name="delete_non_existing_test", 
        start=start_times, 
        end=end_times)

    with pytest.raises(ValueError) as exc_info:
        interval_set.delete_interval(0, 5)
    assert str(exc_info.value) == "Specified interval (0, 5) does not exist."

def test_delete_mismatched_interval():
    start_times = [0.4, 10.6, 20.9]
    end_times = [5.3, 15.8, 25.3]
    
    interval_set = UniqueIntervalSet(
        name="delete_mismatched_test", 
        start=start_times, 
        end=end_times)

    with pytest.raises(ValueError) as exc_info:
        interval_set.delete_interval(10.6, 25.3)
    assert str(exc_info.value) == "Specified interval (10.6, 25.3) does not exist."


'''
Padded interval unit tests.
'''
### Initialization tests ###
def test_init_start_padding_number():
    """Tests if UniqueIntervalSet initializes correctly with start padding passed in as a single integer."""
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]

    start_padding = 1
    
    interval_set = UniqueIntervalSet(
        name="init_start_padding_test", 
        start=start_times, 
        end=end_times, 
        start_padding=start_padding)
    
    assert np.array_equal(interval_set.unpadded_start, start_times)
    assert interval_set.unpadded_end is None
    
    assert np.array_equal(interval_set.start, np.array(start_times) + start_padding)
    assert np.array_equal(interval_set.end, end_times)

    assert len(interval_set.start_padding) == len(interval_set.start)
    assert np.all(interval_set.start_padding == start_padding)
    assert interval_set.end_padding is None

def test_init_end_padding_number():
    """Tests if UniqueIntervalSet initializes correctly with end padding passed in as a single integer."""
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]

    end_padding = -3
    
    interval_set = UniqueIntervalSet(
        name="init_end_padding_test", 
        start=start_times, 
        end=end_times, 
        end_padding=end_padding)
    
    assert interval_set.unpadded_start is None
    assert np.array_equal(interval_set.unpadded_end, end_times)
    
    assert np.array_equal(interval_set.start, start_times)
    assert np.array_equal(interval_set.end, np.array(end_times) + end_padding)

    assert interval_set.start_padding is None
    assert len(interval_set.end_padding) == len(interval_set.end)
    assert np.all(interval_set.end_padding == end_padding)

def test_init_both_padding_number():
    """Tests if UniqueIntervalSet initializes correctly with end padding passed in as a single integer."""
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]
    
    start_padding = -1.5
    end_padding = 2
    interval_set = UniqueIntervalSet(
        name="init_both_padding_test", 
        start=start_times, 
        end=end_times, 
        start_padding=start_padding, 
        end_padding=end_padding)
    
    assert np.array_equal(interval_set.unpadded_start, start_times)
    assert np.array_equal(interval_set.unpadded_end, end_times)
    
    assert np.array_equal(interval_set.start, np.array(start_times) + start_padding)
    assert np.array_equal(interval_set.end, np.array(end_times) + end_padding)

    assert len(interval_set.start_padding) == len(interval_set.start)
    assert np.all(interval_set.start_padding == start_padding)
    assert len(interval_set.end_padding) == len(interval_set.end)
    assert np.all(interval_set.end_padding == end_padding)

def test_init_start_padding_list():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]

    start_padding = [1, 2, 3]
    
    interval_set = UniqueIntervalSet(
        name="start_padding_list_test", 
        start=start_times, 
        end=end_times, 
        start_padding=start_padding)
    
    assert np.array_equal(interval_set.unpadded_start, start_times) 
    assert interval_set.unpadded_end is None
    
    assert np.array_equal(interval_set.start, np.array(start_times) + np.array(start_padding))
    assert np.array_equal(interval_set.end, end_times)

    assert len(interval_set.start_padding) == len(interval_set.start)
    assert np.array_equal(interval_set.start_padding, start_padding)
    assert interval_set.end_padding is None

def test_init_end_padding_ndarray():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]

    end_padding = [-1, -2, -3]
    
    interval_set = UniqueIntervalSet(
        name="start_padding_ndarray_test", 
        start=start_times, 
        end=end_times, 
        end_padding=end_padding)
    
    assert interval_set.unpadded_start is None
    assert np.array_equal(interval_set.unpadded_end, end_times)
    
    assert np.array_equal(interval_set.start, start_times)
    assert np.array_equal(interval_set.end, np.array(end_times) + np.array(end_padding))

    assert interval_set.start_padding is None
    assert np.array_equal(interval_set.end_padding, end_padding)

def test_init_padding_unsupported():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]

    start_padding = (1, 2, 3)

    with pytest.raises(TypeError) as exc_info:
        interval_set = UniqueIntervalSet(
            name="start_padding_unsupported_test", 
            start=start_times, 
            end=end_times, 
            start_padding=start_padding)
    assert str(exc_info.value) == "Unsupported datatype for start padding passed in."

def test_mismatched_start_padding_length():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]

    start_padding = [1]
        
    with pytest.raises(ValueError) as exc_info:
        interval_set = UniqueIntervalSet(
            name="mismatched_start_padlen_test", 
            start=start_times, 
            end=end_times, 
            start_padding=start_padding)
    assert str(exc_info.value) == "Length of start_padding inconsistent with length of start intervals."

def test_mismatched_end_padding_length():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]

    end_padding = [1, 2, 3, 4]
        
    with pytest.raises(ValueError) as exc_info:
        interval_set = UniqueIntervalSet(
            name="mismatched_end_padlen_test", 
            start=start_times, 
            end=end_times, 
            end_padding=end_padding)
    assert str(exc_info.value) == "Length of end_padding inconsistent with length of end intervals."

def test_init_padding_with_overlaps():
    # Given some overlapping intervals
    start_times = [0, 5, 10, 15, 20]
    end_times = [12, 17, 22, 27, 30]

    start_padding = 2.5
    # When we create a UniqueIntervalSet with them
    interval_set = UniqueIntervalSet(
        name="padding_overlap_test", 
        start=start_times, 
        end=end_times, 
        start_padding=start_padding)

    # Then it should merge overlapping intervals
    # Here, all intervals overlap, so they should be merged into one
    # Note: Pynapple may output warning that "some starts precede the previous end", and so it is joining these intervals
    assert len(interval_set.start) == len(interval_set.end) == 1
    assert interval_set.start[0] == 2.5
    assert interval_set.end[0] == 30
    assert interval_set.unpadded_start[0] == 0
    assert interval_set.unpadded_end is None
    assert len(interval_set.start_padding) == len(interval_set.start)
    assert all(interval_set.start_padding == start_padding)
    assert interval_set.end_padding is None

def test_init_padding_with_non_unique():
    # Given some non-unique intervals
    start_times = [0, 15, 0]
    end_times = [12, 20, 12]

    end_padding = 2.5

    # When we create a UniqueIntervalSet with them
    interval_set = UniqueIntervalSet(
        name="padding_non_unique_test", 
        start=start_times, 
        end=end_times, 
        end_padding=end_padding)

    # Then it should merge non-unique intervals
    assert np.array_equal(interval_set.start, [0, 15])
    assert np.array_equal(interval_set.end, np.array([12, 20]) + end_padding)

    assert interval_set.unpadded_start is None
    assert np.array_equal(interval_set.unpadded_end, [12, 20])

    assert interval_set.start_padding is None
    assert len(interval_set.end_padding) == len(interval_set.end)
    assert all(interval_set.end_padding == end_padding)
    
def test_padding_illegal_start_end_times():
    start_times = [2, 5, 20]
    end_times = [1, 4, 25]

    start_padding = -4.9
    
    # IntervalSet will drop intervals with start times preceding end times
    interval_set = UniqueIntervalSet(
        name="padding_illegal_times_test", 
        start=start_times, 
        end=end_times, 
        start_padding=start_padding)

    assert len(interval_set.start) == len(interval_set.end) == 1
    assert interval_set.start[0] == 15.1
    assert interval_set.end[0] == 25
    assert interval_set.unpadded_start[0] == 20
    assert interval_set.unpadded_end is None
    assert all(interval_set.start_padding == start_padding)
    assert len(interval_set.start_padding) == len(interval_set.start)
    assert interval_set.end_padding is None

### Add interval tests ###
def test_padding_add_interval():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]
    
    end_padding = 2
    interval_set = UniqueIntervalSet(
        name="padding_add_test", 
        start=start_times, 
        end=end_times, 
        end_padding=end_padding)

    interval_set.add_interval(30, 32, end_pad=end_padding)
    assert np.array_equal(interval_set.start, [0, 10, 20, 30])
    assert np.array_equal(interval_set.end, np.array([5, 15, 25, 32]) + end_padding)

    assert interval_set.unpadded_start is None
    assert np.array_equal(interval_set.unpadded_end, [5, 15, 25, 32])

    assert interval_set.start_padding is None
    assert len(interval_set.end_padding) == len(interval_set.end)
    assert all(interval_set.end_padding == end_padding)

def test_padding_add_without_pad():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]
    
    end_padding = 2
    interval_set = UniqueIntervalSet(
        name="padding_add_test", 
        start=start_times, 
        end=end_times, 
        end_padding=end_padding)

    with pytest.raises(ValueError) as exc_info:
        interval_set.add_interval(30, 32)
    assert str(exc_info.value) == "End padding for new interval is missing"

def test_missing_padding_add_pad():
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]
    
    end_padding = 2
    interval_set = UniqueIntervalSet(
        name="padding_add_test", 
        start=start_times, 
        end=end_times, 
        end_padding=end_padding)

    with pytest.raises(ValueError) as exc_info:
        interval_set.add_interval(30, 32, start_pad=2)
    assert str(exc_info.value) == "IntervalSet does not have start padding. Cannot pass in start padding for new interval."

# ### Delete interval tests ###
def test_padding_delete_existing_interval():
    start_times = [0.4, 10.6, 20.9]
    end_times = [5.3, 15.8, 25.3]

    start_padding = -2
    
    interval_set = UniqueIntervalSet(
        name="padding_delete_existing_test", 
        start=start_times, 
        end=end_times, 
        start_padding=start_padding)

    interval_set.delete_interval(10.6, 15.8)
    assert np.array_equal(interval_set.start, np.array([0.4, 20.9]) + start_padding)
    assert np.array_equal(interval_set.end, [5.3, 25.3])

    assert np.array_equal(interval_set.unpadded_start, [0.4, 20.9])
    assert interval_set.unpadded_end is None

    assert len(interval_set.start_padding) == len(interval_set.start)
    assert np.all(interval_set.start_padding == start_padding)
    assert interval_set.end_padding is None

def test_padding_delete_mismatched_interval():
    start_times = [0.4, 10.6, 20.9]
    end_times = [5.3, 15.8, 25.3]

    end_padding = 1
    
    interval_set = UniqueIntervalSet(
        name="padding_delete_mismatched_test", 
        start=start_times, 
        end=end_times, 
        end_padding=end_padding)

    with pytest.raises(ValueError) as exc_info:
        interval_set.delete_interval(10.6, 16.8)
    assert str(exc_info.value) == "Specified interval (10.6, 16.8) does not exist."
