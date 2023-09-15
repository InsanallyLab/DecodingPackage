import pytest
from decoder.core.event_set import EventSet


def test_eventSet_initialization():
    """Test if EventSet initializes correctly with expected data"""
    event_name = "test"
    timestamps = [5, 10, 15]
    labels = ["ex1", "ex2", "ex3"]
    event_set = EventSet(event_name, timestamps, labels)
    assert len(event_set.events) == 3
