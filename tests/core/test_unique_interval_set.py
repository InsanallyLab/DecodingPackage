# tests/core/test_unique_interval_set.py

import pytest
from decoder.core.unique_interval_set import UniqueIntervalSet

def test_initialization_without_overlaps():
    """Test if UniqueIntervalSet initializes correctly without overlapping intervals."""
    start_times = [0, 10, 20]
    end_times = [5, 15, 25]
    
    uis = UniqueIntervalSet(name="test", start=start_times, end=end_times)
    
    assert len(uis) == 3, "Unexpected number of intervals in UniqueIntervalSet."

def test_initialization_with_overlaps():
    # Given some overlapping intervals
    start_times = [0, 5, 10, 15, 20]
    end_times = [12, 17, 22, 27, 30]

    # When we create a UniqueIntervalSet with them
    interval_set = UniqueIntervalSet(start=start_times, end=end_times)

    # Then it should merge overlapping intervals
    # Here, all intervals overlap, so they should be merged into one
    assert len(interval_set) == 1
    assert interval_set["start"].iloc[0] == 0
    assert interval_set["end"].iloc[0] == 30

def test_empty_initialization():
    """Test if UniqueIntervalSet initializes correctly with empty start and end lists."""
    uis = UniqueIntervalSet(name="test", start=[], end=[])    
    assert uis.empty, "UniqueIntervalSet should be empty."

def test_mismatched_start_end_lengths():
    """Test if ValueError is raised when start and end lists are of different lengths."""
    with pytest.raises(RuntimeError):
        uis = UniqueIntervalSet(name="test", start=[0, 10], end=[5])

# Add more tests as necessary
