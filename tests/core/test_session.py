from decoder.core.unique_interval_set import UniqueIntervalSet
import pytest
import pynapple as nap
from decoder.core.session import Session
import numpy as np

### Initialization tests ###
def test_session_init():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

    assert np.array_equal(session.spike_train.t, spike_train)
    assert len(session.interval_sets) == 1
    assert np.array_equal(session.interval_sets[iset_name].start, start)
    assert np.array_equal(session.interval_sets[iset_name].end, end)
    assert len(session.interval_sets) == 1
    assert np.array_equal(session.event_sets[eset_name].t, event_times)
    assert np.array_equal(session.event_sets[eset_name].values, event_labels)

def test_session_init_spikes_ndarray():
    spike_times = np.array([0.7, 1.4, 2.8, 3.4])
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

    assert np.array_equal(session.spike_train.t, spike_times)
    assert len(session.interval_sets) == 1
    assert np.array_equal(session.interval_sets[iset_name].start, start)
    assert np.array_equal(session.interval_sets[iset_name].end, end)
    assert len(session.interval_sets) == 1
    assert np.array_equal(session.event_sets[eset_name].t, event_times)
    assert np.array_equal(session.event_sets[eset_name].values, event_labels)

def test_session_init_unique_interval_set():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    uiset = UniqueIntervalSet(start=start, end=end, name=iset_name)
    interval_sets = {uiset.name : uiset}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

    assert np.array_equal(session.spike_train.t, spike_train)
    assert len(session.interval_sets) == 1
    assert np.array_equal(session.interval_sets[iset_name].start, start)
    assert np.array_equal(session.interval_sets[iset_name].end, end)
    assert len(session.interval_sets) == 1
    assert np.array_equal(session.event_sets[eset_name].t, event_times)
    assert np.array_equal(session.event_sets[eset_name].values, event_labels)

def test_session_init_iset_wrong_type():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    interval_sets = [start, end]

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    with pytest.raises(TypeError) as exc_info:
        session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)
    assert str(exc_info.value) == "interval_sets must be a dict."

def test_session_init_eset_wrong_type():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_sets = [0.5, 2.6]

    with pytest.raises(TypeError) as exc_info:
        session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)
    assert str(exc_info.value) == "event_sets must be a dict."

def test_session_init_iset_wrong_key_type():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {0: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    with pytest.raises(TypeError) as exc_info:
        session = Session(
            spike_times=spike_times, 
            interval_sets=interval_sets, 
            event_sets=event_sets)
    assert str(exc_info.value) == "All keys in interval_sets dict must be strings."

def test_session_init_eset_wrong_value_type():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    eset_name = "lick times"
    event_sets = {eset_name: event_times}

    with pytest.raises(TypeError) as exc_info:
        session = Session(
            spike_times=spike_times,
            interval_sets=interval_sets, 
            event_sets=event_sets)
    assert str(exc_info.value) == "All values in event_sets dict must be nap.Tsd."

### slice_spikes_by_intervals tests ###
def test_slice_spikes_incorrect_iset_name():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

    with pytest.raises(KeyError) as exc_info:
        session.slice_spikes_by_intervals(iset_name="trials")
    # str(KeyError) adds extra quotes ('') in error string
    assert str(exc_info.value) == "'Interval set name does not exist'"

def test_slice_spikes():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)
        
    session.slice_spikes_by_intervals(iset_name=iset_name)

    interval_to_spikes = session.interval_to_spikes[iset_name]
    assert len(interval_to_spikes) == 2
    assert np.array_equal(interval_to_spikes[(0, 2)].t, [0.7, 1.4])
    assert np.array_equal(interval_to_spikes[(2.5, 4)].t, [2.8, 3.4])

    assert len(session.locked_interval_to_spikes) == 0

def test_slice_spikes_outside_intervals():
    spike_train = [0.7, 1.4, 2.8, 3.4, 4.2, 5.8]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

    session.slice_spikes_by_intervals(iset_name=iset_name)

    interval_to_spikes = session.interval_to_spikes[iset_name]
    assert len(interval_to_spikes) == 2
    assert np.array_equal(interval_to_spikes[(0, 2)].t, [0.7, 1.4])
    assert np.array_equal(interval_to_spikes[(2.5, 4)].t, [2.8, 3.4])

    assert len(session.locked_interval_to_spikes) == 0

def test_slice_spikes_empty_intervals():
    spike_train = [0.7, 1.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

    session.slice_spikes_by_intervals(iset_name=iset_name)

    interval_to_spikes = session.interval_to_spikes[iset_name]
    assert len(interval_to_spikes) == 2
    assert np.array_equal(interval_to_spikes[(0, 2)].t, [0.7, 1.4])
    assert np.array_equal(interval_to_spikes[(2.5, 4)].t, [])

    assert len(session.locked_interval_to_spikes) == 0

### time_lock_to_interval tests ###
@pytest.fixture
def mock_session():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    return Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

def test_time_lock_interval_pre_slicing(mock_session):
    with pytest.raises(KeyError) as exc_info:
        mock_session.time_lock_to_interval(
            iset_name="trial times", 
            lock_point="start")
    # str(KeyError) adds extra quotes ('') in error string
    assert str(exc_info.value) == "'Spikes have not been mapped to this interval set. Run slice_spikes_by_intervals first.'"

def test_time_lock_interval_wrong_lockpoint(mock_session):
    mock_session.slice_spikes_by_intervals(iset_name="trial times")
    with pytest.raises(ValueError) as exc_info:
        mock_session.time_lock_to_interval(
            iset_name="trial times", 
            lock_point="event")
    assert str(exc_info.value) == "lock_point should be either 'start' or 'end'"

def test_time_lock_interval_wrong_iset(mock_session):
    mock_session.slice_spikes_by_intervals(iset_name="trial times")
    with pytest.raises(KeyError) as exc_info:
        mock_session.time_lock_to_interval(
            iset_name="trials", 
            lock_point="start")
    assert str(exc_info.value) == "'Interval set name does not exist'"

def test_time_lock_interval_start(mock_session):
    mock_session.slice_spikes_by_intervals(iset_name="trial times")
    mock_session.time_lock_to_interval(
        iset_name="trial times", 
        lock_point="start")

    locked_interval_to_spikes = mock_session.locked_interval_to_spikes[("start", "trial times")]
    assert len(locked_interval_to_spikes) == 2
    assert np.array_equal(locked_interval_to_spikes[(0, 2)].t, [0.7, 1.4])
    assert np.array_equal(locked_interval_to_spikes[(2.5, 4)].t, [0.3, 0.9])

def test_time_lock_interval_end(mock_session):
    mock_session.slice_spikes_by_intervals(iset_name="trial times")
    mock_session.time_lock_to_interval(
        iset_name="trial times", 
        lock_point="end")

    locked_interval_to_spikes = mock_session.locked_interval_to_spikes[("end", "trial times")]
    assert len(locked_interval_to_spikes) == 2
    assert np.array_equal(locked_interval_to_spikes[(0, 2)].t, [-1.3, -0.6])
    assert np.array_equal(locked_interval_to_spikes[(2.5, 4)].t, [-1.2, -0.6])


### _match_event_to_interval tests ###
def test_match_one_event_to_interval():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 2.6]
    event_labels = ["lick", "lick"]
    event_set = nap.Tsd(t=event_times, d=event_labels)
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

    session.slice_spikes_by_intervals(iset_name=iset_name)

    matched_events = session._match_event_to_interval(
        event_set=event_set, 
        iset_name=iset_name)
    assert len(matched_events) == 2
    assert matched_events[(0, 2)] == 0.5
    assert matched_events[(2.5, 4)] == 2.6

def test_match_zero_event_to_interval():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 5.2]
    event_labels = ["lick", "lick"]
    event_set = nap.Tsd(t=event_times, d=event_labels)
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

    session.slice_spikes_by_intervals(iset_name=iset_name)

    with pytest.raises(ValueError) as exc_info:
        matched_events = session._match_event_to_interval(
        event_set=event_set, 
        iset_name=iset_name)
    assert str(exc_info.value) == "Interval (2.5, 4.0) does not match exactly one event in event_set."

def test_match_multiple_events_to_interval():
    spike_train = [0.7, 1.4, 2.8, 3.4]
    spike_times = nap.Ts(t=spike_train)
    
    start = [0, 2.5]
    end = [2, 4]
    iset_name = "trial times"
    interval_sets = {iset_name: nap.IntervalSet(start=start, end=end)}

    event_times = [0.5, 1.0]
    event_labels = ["lick", "lick"]
    event_set = nap.Tsd(t=event_times, d=event_labels)
    eset_name = "lick times"
    event_sets = {eset_name: nap.Tsd(t=event_times, d=event_labels)}

    session = Session(
        spike_times=spike_times, 
        interval_sets=interval_sets, 
        event_sets=event_sets)

    session.slice_spikes_by_intervals(iset_name=iset_name)

    with pytest.raises(ValueError) as exc_info:
        matched_events = session._match_event_to_interval(
        event_set=event_set, 
        iset_name=iset_name)
    assert str(exc_info.value) == "Interval (0.0, 2.0) does not match exactly one event in event_set."

### time_lock_to_event tests ###
def test_time_lock_event_wrong_iset(mock_session):
    mock_session.slice_spikes_by_intervals(iset_name="trial times")

    with pytest.raises(KeyError) as exc_info:
        mock_session.time_lock_to_event(
        eset_name="lick times", 
        iset_name="trials")
    # str(KeyError) adds extra quotes ('') in error string
    assert str(exc_info.value) == "'Interval set name does not exist'"

def test_time_lock_event_wrong_eset(mock_session):
    mock_session.slice_spikes_by_intervals(iset_name="trial times")

    with pytest.raises(KeyError) as exc_info:
        mock_session.time_lock_to_event(
        eset_name="licks", 
        iset_name="trial times")
    # str(KeyError) adds extra quotes ('') in error string
    assert str(exc_info.value) == "'Event set name passed in as lock point does not exist'"

def test_time_lock_event_pre_slicing(mock_session):
    with pytest.raises(KeyError) as exc_info:
        mock_session.time_lock_to_event(
        eset_name="lick times", 
        iset_name="trial times")
    # str(KeyError) adds extra quotes ('') in error string
    assert str(exc_info.value) == "'Spikes have not been mapped to this interval set. Run slice_spikes_by_intervals first.'"

def test_time_lock_event(mock_session):
    iset_name = "trial times"
    eset_name = "lick times"
    mock_session.slice_spikes_by_intervals(iset_name=iset_name)
    mock_session.time_lock_to_event(eset_name=eset_name, iset_name=iset_name)

    locked_interval_to_spikes = mock_session.locked_interval_to_spikes[(eset_name, iset_name)]
    assert len(locked_interval_to_spikes) == 2
    assert np.array_equal(locked_interval_to_spikes[(0, 2)].t, [0.2, 0.9])
    assert np.array_equal(locked_interval_to_spikes[(2.5, 4)].t, [0.2, 0.8])


### compute_log_ISIs tests ###
def test_log_ISIs(mock_session):
    iset_name = "trial times"
    mock_session.compute_log_ISIs(iset_name=iset_name)

    assert len(mock_session.iset_to_log_ISIs) == 1
    log_ISIs = mock_session.iset_to_log_ISIs[iset_name]

    default_scaling_factor = 1000
    expected_ISIs = np.array([[0.7], [0.6]]) * default_scaling_factor
    expected_log_ISIs = np.log10(expected_ISIs)
    assert np.array_equal(log_ISIs, expected_log_ISIs)

    assert len(mock_session.locked_iset_to_log_ISIs) == 0

def test_log_ISIs_with_time_lock_event(mock_session):
    iset_name = "trial times"
    eset_name = "lick times"
    mock_session.compute_log_ISIs(iset_name=iset_name, lock_point=eset_name)

    assert len(mock_session.locked_iset_to_log_ISIs) == 1
    log_ISIs = mock_session.locked_iset_to_log_ISIs[(eset_name, iset_name)]

    default_scaling_factor = 1000
    expected_ISIs = np.array([[0.7], [0.6]]) * default_scaling_factor
    expected_log_ISIs = np.log10(expected_ISIs)
    assert np.array_equal(log_ISIs, expected_log_ISIs)

    assert len(mock_session.iset_to_log_ISIs) == 0
