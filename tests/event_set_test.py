import pytest
from decoder.core.event_set import EventSet


def create_event_set(timestamps, labels, possible_values=None):
    event_name = "test"
    return EventSet(event_name, timestamps, labels, possible_values)


def test_eventSet_initialization():
    """Test if EventSet initializes correctly with expected data"""
    event_set = create_event_set([5, 10, 15], ["ex1", "ex2", "ex3"])
    assert len(event_set.events) == 3


def test_eventSet_timestamps_shorter_than_labels():
    """Test if EventSet raises a ValueError when there are less timestamps than labels"""
    with pytest.raises(ValueError):
        event_set = create_event_set([5, 10], ["ex1", "ex2", "ex3"])


def test_eventSet_labels_shorter_than_timestamps():
    """Test if EventSet raises a ValueError when there are less labels than timestamps"""
    with pytest.raises(ValueError):
        event_set = create_event_set([5, 10, 15], ["ex1", "ex2"])


def test_eventSet_invalid_label():
    """Test if EventSet raises a ValueError when there are invalid labels (i.e. labels not in possible values)"""
    with pytest.raises(ValueError):
        event_set = create_event_set(
            [5, 10, 15], ["ex1", "ex2", "ex3"], ["A", "B", "C"])


def test_add_valid_event():
    """Test if EventSet adds valid events"""
    event_set = create_event_set(
        [5, 10, 15], ["ex1", "ex2", "ex3"], ["ex1", "ex2", "ex3", "ex4"])
    timestamps = [5, 10, 15]
    new_timestamp = 20
    new_label = "ex4"

    event_set.add_event(new_timestamp, new_label)

    assert event_set.events[new_timestamp] == new_label
    assert event_set.get_sorted_events() == sorted(
        timestamps + [new_timestamp])


def test_add_event_with_invalid_label():
    """Test if EventSet raises ValueError when adding an event with an invalid label."""
    event_set = create_event_set(
        [5, 10, 15], ["ex1", "ex2", "ex3"], ["ex1", "ex2", "ex3"])

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
    event_set = create_event_set([5, 10, 15], ["ex1", "ex2", "ex3"])
    timestamp_to_delete = 10

    event_set.delete_event(timestamp_to_delete)

    assert timestamp_to_delete not in event_set.events
    assert timestamp_to_delete not in event_set.get_sorted_events()


def test_delete_event_with_invalid_timestamp():
    """Test if EventSet raises KeyError when deleting an event with a non-existent timestamp"""
    event_set = create_event_set([5, 10, 15], ["ex1", "ex2", "ex3"])
    invalid_timestamp = 50

    with pytest.raises(KeyError):
        event_set.delete_event(invalid_timestamp)


def test_sort_events():
    """Test if EventSet sorts events"""
    timestamps = [15, 20, 5, 10]
    event_set = create_event_set([15, 20, 5, 10], ["ex3", "ex4", "ex1", "ex2"])

    sorted_timestamps = event_set.get_sorted_events()

    assert sorted_timestamps == sorted(timestamps)
