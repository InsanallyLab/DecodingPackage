import pytest
from decoder.core.event_set import EventSet


def test_eventSet_initialization():
    """Test if EventSet initializes correctly with expected data"""
    event_name = "test"
    timestamps = [5, 10, 15]
    labels = ["ex1", "ex2", "ex3"]
    event_set = EventSet(event_name, timestamps, labels)
    assert len(event_set.events) == 3


def test_eventSet_timestamps_shorter_than_labels():
    """Test if EventSet raises a ValueError when there are less timestamps than labels"""
    event_name = "test"
    timestamps = [5, 10]
    labels = ["ex1", "ex2", "ex3"]
    with pytest.raises(ValueError):
        event_set = EventSet(event_name, timestamps, labels)


def test_eventSet_labels_shorter_than_timestamps():
    """Test if EventSet raises a ValueError when there are less labels than timestamps"""
    event_name = "test"
    timestamps = [5, 10, 15]
    labels = ["ex1", "ex2"]
    with pytest.raises(ValueError):
        event_set = EventSet(event_name, timestamps, labels)


def test_eventSet_invalid_label():
    """Test if EventSet raises a ValueError when there are invalid labels (i.e. labels not in possible values)"""
    event_name = "test"
    timestamps = [5, 10, 15]
    labels = ["ex1", "ex2", "ex3"]
    possible_values = ["A", "B", "C"]
    with pytest.raises(ValueError):
        event_set = EventSet(event_name, timestamps, labels, possible_values)


def test_eventSet_invalid_timepoint():
    """Test if EventSet raises a ValueError when there are invalid timepoints"""
    event_name = "test"
    timestamps = [5, 10, 15]
    labels = ["ex1", "ex2", "ex3"]
    invalid_timepoint = [5]
    with pytest.raises(ValueError):
        event_set = EventSet.add_event(event_name, timestamps, labels,  # added add_event last
                                       invalid_timepoint)
