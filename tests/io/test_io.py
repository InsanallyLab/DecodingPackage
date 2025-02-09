from decoder.core.ndecoder import NDecoder
from decoder.core.session import Session
from decoder.core.unique_interval_set import UniqueIntervalSet
import pickle
import numpy as np
from typing import Optional
import pynapple as nap
from decoder.io.pickle_to_nwb import pickle_to_nwb
from decoder.io.utils import convert_units

def test_pickle_to_pynapple(
    pickle_path: str, 
    nwb_save_path: str, 
    iset_name: str,
    eset_name: str, 
    input_time_units: Optional[str] = None) -> tuple[Session, nap.Tsd]:
    '''
    Tests loading pickle file data using NeuroConv & Pynapple. 

    Parameters
    ----------
    pickle_path : str
        Path to the pickle file that needs to be loaded.
    nwb_save_path : str
        Path where the generated nwb file should be saved. 
    iset_name : str
        Name for the interval set (with trial start and end times) loaded from 
        the pickle file. 
    eset_name : str
        Name for the event set loaded from the pickle file. 
    input_time_units : str, optional
        If input data is not in seconds, pass in the time unit used. The function
        convert_units() will be called so that all created Pynapple objects are
        in the correct time unit.  

    Returns 
    -------
    session : Session
        The session object generated with all the input data.
    trial_conditions : nap.Tsd
        The condition for each trial.
    '''
    pickle_to_nwb(pickle_path=pickle_path, nwb_save_path=nwb_save_path)

    data = nap.load_file(nwb_save_path)

    interval_set = data["trial_times"] # IntervalSet
    spike_times = data["spikes"] # Ts
    lick_series = data["licks"] # Tsd
    trial_conditions = data["trial_conditions"] # Tsd
    
    # Allows user to change time units if needed
    if input_time_units is not None:
        interval_set = convert_units(interval_set, input_time_units)
        spike_times = convert_units(spike_times, input_time_units)
        lick_series = convert_units(lick_series, input_time_units)

    # If using IntervalSet 
    interval_sets = {iset_name: interval_set}
    # If using padded UniqueIntervalSet
    unique_iset = UniqueIntervalSet(name=iset_name, start=interval_set.start, end=interval_set.end, start_padding=-4)
    unique_isets = {unique_iset.name : unique_iset}

    event_sets = {eset_name: lick_series}

    session = Session(spike_times=spike_times, interval_sets=unique_isets, event_sets=event_sets)

    return session, trial_conditions

def test_saving_loading_ndecoder(n_decoder: NDecoder, file_path: str): 
    '''
    Tests saving and then re-loading NDecoder object from a pickle file. 

    Parameters
    ----------
    n_decoder : NDecoder
        NDecoder object to save and then re-load.
    file_path : str
        Path to store log ISIs .npz file at.
    '''
    n_decoder.save_as_pickle(file_path)
    file = open(file_path, 'rb')
    loaded_ndecoder = pickle.load(file)
    file.close()

    assert n_decoder.bw == loaded_ndecoder.bw
    assert n_decoder.min_ISIs == loaded_ndecoder.min_ISIs
    assert np.array_equal(n_decoder.conditions, loaded_ndecoder.conditions)

    assert n_decoder.model.all.pdf(-1.9) == loaded_ndecoder.model.all.pdf(-1.9)
    assert n_decoder.model.all.inv_cdf(0.4) == loaded_ndecoder.model.all.inv_cdf(0.4)
    
    for cond in n_decoder.conditions:
        assert n_decoder.model.conds[cond].prior_0 == loaded_ndecoder.model.conds[cond].prior_0
        assert n_decoder.model.conds[cond].prior_empty == loaded_ndecoder.model.conds[cond].prior_empty

        assert n_decoder.model.conds[cond].pdf(3) == loaded_ndecoder.model.conds[cond].pdf(3)
        assert n_decoder.model.conds[cond].inv_cdf(0.65) == loaded_ndecoder.model.conds[cond].inv_cdf(0.65)

def test_saving_loading_log_ISIs(
    session: Session, 
    iset_name: str, 
    file_path: str, 
    lock_point: Optional[str] = None):
    '''
    Tests saving and then re-loading log ISIs for a given interval set from a 
    Session object.
    If a lock point is provided, the Session object will first time lock the 
    interval set before computing log ISIs.

    Parameters
    ----------
    session : Session
        Session object used to save log ISIs.
    iset_name : str
        Name of the interval set to save log ISIs for. 
    file_path : str
        Path to store log ISIs .npz file at.
    lock_point : str, optional
        Name of the lock point (an event set, or the start/end of the 
        interval) to time lock to.
    '''
    if lock_point is not None:
        log_ISIs_train = session.locked_iset_to_log_ISIs[(lock_point, iset_name)]
    else:
        log_ISIs_train = session.iset_to_log_ISIs[iset_name]

    session.save_log_ISIs(iset_name=iset_name, file_path=file_path, lock_point=lock_point)
    data = np.load(file_path, allow_pickle=True)

    if lock_point is not None:
        loaded_log_ISIs_train = data["locked_log_ISIs"]
    else: 
        loaded_log_ISIs_train = data["log_ISIs"]
    data.close()

    assert len(loaded_log_ISIs_train) == len(log_ISIs_train)
    
    for trial_idx in range(len(log_ISIs_train)):
        log_ISIs = log_ISIs_train[trial_idx]
        loaded_log_ISIs = loaded_log_ISIs_train[trial_idx]
        assert np.array_equal(log_ISIs, loaded_log_ISIs)

def test_saving_loading_spikes(
    session: Session, 
    iset_name: str, 
    file_path: str, 
    lock_point: Optional[str] = None):
    '''
    Tests saving and then re-loading spike trains for a given interval set from 
    a Session object.
    If a lock point is provided, it saves the time-locked spike train instead.

    Parameters
    ----------
    session : Session
        Session object used to save spike train.
    iset_name : str
        Name of the interval set to save spike train for. 
    file_path : str
        Path to store spike train .npz file at.
    lock_point : str, optional
        Name of the lock point (an event set, or the start/end of the 
        interval) to time lock to.
    '''
    if lock_point is not None:
        spikes = session.locked_iset_to_spikes[(lock_point, iset_name)]
    else:
        spikes = session.iset_to_spikes[iset_name]

    session.save_spikes(iset_name=iset_name, file_path=file_path, lock_point=lock_point)
    loaded_spikes = nap.load_file(file_path)
    
    assert np.array_equal(spikes.t, loaded_spikes.t)
    

