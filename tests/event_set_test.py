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


def test_add_valid_event():
    """Test if EventSet adds valid events"""
    event_name = "test"
    timestamps = [5, 10, 15]
    labels = ["ex1", "ex2", "ex3"]
    event_set = EventSet(event_name, timestamps, labels,
                         possible_values=["ex1", "ex2", "ex3", "ex4"])
    new_timestamp = 20
    new_label = "ex4"

    event_set.add_event(new_timestamp, new_label)

    assert event_set.events[new_timestamp] == new_label
    assert event_set.get_sorted_events() == sorted(
        timestamps + [new_timestamp])


def test_add_event_with_invalid_label():
    """Test if EventSet raises ValueError when adding an event with an invalid label."""
    event_name = "test"
    timestamps = [5, 10, 15]
    labels = ["ex1", "ex2", "ex3"]
    event_set = EventSet(event_name, timestamps, labels,
                         possible_values=["ex1", "ex2", "ex3"])
    new_timestamp = 20
    new_label = "ex4"

    with pytest.raises(ValueError):
        event_set.add_event(new_timestamp, new_label)


# TODO: Test that a ValueError is raised when an event with an existing timestamp is added
# def test_add_event_with_existing_timestamp():
#     """Test if EventSet raises ValueError when adding an event with an invalid timestamp."""
#     event_name = "test"
#     timestamps = [5, 10, 15]
#     labels = ["ex1", "ex2", "ex3"]
#     event_set = EventSet(event_name, timestamps, labels)

#     existing_timestamp = 100
#     new_label = "ex4"

#     with pytest.raises(ValueError):
#         event_set.add_event(existing_timestamp, new_label)


def test_delete_event():
    """Test if EventSet deletes valid events"""
    event_name = "test"
    timestamps = [5, 10, 15]
    labels = ["ex1", "ex2", "ex3"]
    event_set = EventSet(event_name, timestamps, labels)
    timestamp_to_delete = 10

    event_set.delete_event(timestamp_to_delete)

    assert timestamp_to_delete not in event_set.events
    assert timestamp_to_delete not in event_set.get_sorted_events()


def test_delete_event_with_invalid_timestamp():
    """Test if EventSet raises KeyError when deleting an event with a non-existent timestamp"""
    event_name = "test"
    timestamps = [5, 10, 15]
    labels = ["ex1", "ex2", "ex3"]
    event_set = EventSet(event_name, timestamps, labels)
    invalid_timestamp = 50

    with pytest.raises(KeyError):
        event_set.delete_event(invalid_timestamp)
