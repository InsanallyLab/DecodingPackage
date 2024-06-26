from core.ndecoder import NDecoder
from core.session import Session
from core.bandwidth import Bandwidth
from core.unique_interval_set import UniqueIntervalSet
import numpy as np
import pynapple as nap 
import pickle
from decoder.io.pickle_to_nwb import pickle_to_nwb
from typing import Optional
from numpy.typing import NDArray

def convert_units(nap_object, input_units: str = "s"):
    """
    Reinitializes all Pynapple input objects with the correct time units.
    Pynapple assumes that all input data is passed in as seconds. If this
    is not the case, use this function to convert all Pynapple objects 
    to the correct time unit.

    Parameters
    ----------
    nap_object : Ts, Tsd, or IntervalSet 
        Pynapple object to be converted to the correct time unit.
    units : str, optional
        ('us', 'ms', 's' [default])

    Returns 
    -------
    The Pynapple object converted to the correct time unit.
    """
    if isinstance(nap_object, nap.core.time_series.Ts):
        return nap.Ts(t=nap_object.t, time_units=input_units)

    if isinstance(nap_object, nap.core.time_series.Tsd):
        return nap.Tsd(t=nap_object.t, d=nap_object.d, time_units=input_units)
    
    if isinstance(nap_object, nap.core.interval_set.IntervalSet):
        return nap.IntervalSet(start=nap_object.start, end=nap_object.end, time_units=input_units)

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

    # CHANGED: Event data stored as Tsd instead of EventSet
    event_sets = {eset_name: lick_series}

    session = Session(spike_times=spike_times, interval_sets=unique_isets, event_sets=event_sets)

    return session, trial_conditions

def test_saving_loading_log_ISIs(
    session: Session, 
    iset_name: str, 
    file_path: str, 
    lock_point: Optional[str] = None) -> NDArray:
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

    Returns 
    -------
    log_ISIs_train : numpy.ndarray
        The saved and re-loaded log ISIs for the interval set.
    '''
    session.save_log_ISIs(iset_name=iset_name, file_path=file_path, lock_point=lock_point)
    data = np.load(file_path, allow_pickle=True)
    log_ISIs_train = data["locked_log_ISIs"]
    data.close()
    return log_ISIs_train

def test_saving_loading_ndecoder(n_decoder : NDecoder ,file_path : str) -> NDecoder: 
    '''
    Tests saving and then re-loading NDecoder object from a pickle file. 

    Parameters
    ----------
    n_decoder : NDecoder
        NDecoder object to save and then re-load.
    file_path : str
        Path to store log ISIs .npz file at.

    Returns 
    -------
    n_decoder : NDecoder
        The saved and re-loaded NDecoder object.
    '''
    n_decoder.save_as_pickle(file_path)
    file = open(file_path, 'rb')
    n_decoder = pickle.load(file)
    file.close()
    return n_decoder
    
def test_workflow():
    '''
    Function demonstrating minimal working example.
    Uses Neuroconv, Pynapple to load pickle file data as Pynapple objects.
    Runs decoding algorithm on Pynapple objects, generates decoder, saves 
    decoder as pickle file.  
    '''
    train_file = 'AE_238_1_AC.pickle'
    nwb_save_path = "./train_trials.nwb" 

    train_session, conditions_train = test_pickle_to_pynapple(
        pickle_path=train_file, 
        nwb_save_path=nwb_save_path,
        iset_name="train trials",
        eset_name="lick events")

    print("Done making train Session object")

    log_ISIs_train = train_session.compute_log_ISIs(iset_name="train trials", lock_point="start")
    print("Done computing train log_ISIs")

    print("Saving and then loading log ISIs from local npz file")
    log_ISIs_train = test_saving_loading_log_ISIs(
        session=train_session,
        iset_name="train trials",
        file_path="log_ISIs.npz",
        lock_point="start")

    log_ISIs_concat = np.concatenate(log_ISIs_train)
    bandwidth_folds = 10    # from Insanally paper
    # TODO: grid_search is taking too long
    # kde_bandwidth = Bandwidth.sklearn_grid_search_bw(log_ISIs_concat, bandwidth_folds)
    kde_bandwidth = 0.20
    print("KDE bandwidth: ", kde_bandwidth)

    min_ISIs = 0 
    possible_conditions = ["target", "non-target"]
    n_decoder = NDecoder(kde_bandwidth, min_ISIs, possible_conditions)
    print("NDecoder made")

    reps = 2    # 124 in Insanally paper
    K_fold_num = 2  # 10 in Insanally paper
    accuracy_per_fold = []
    frac_emptyISIs_per_fold = []

    conditions_train = np.asarray(conditions_train.values, dtype=str)

    for rep in (range(int(reps/K_fold_num))):
        train_validate_pairs = n_decoder.generate_stratified_K_folds(log_ISIs_train, conditions_train, K_fold_num)
        print("Generated stratified folds")

        for K, (train_data, validate_data) in enumerate(train_validate_pairs):
            train_X, train_y = train_data
            n_decoder.fit(train_X, train_y)
            print("Model fit on fold: %d, rep: %d" %(K, rep))

            # Compute fold validation accuracy 
            validate_X, validate_y = validate_data
            accuracy, frac_empty = n_decoder.calculate_accuracy(validate_X, validate_y)
            print("Calculated validation accuracy: %f, fraction empty ISIs: %f" %(accuracy, frac_empty))
            accuracy_per_fold.append(accuracy)
            frac_emptyISIs_per_fold.append(frac_empty)

    print("Model fitting complete")
    mean_accuracy = np.nanmean(accuracy_per_fold)
    mean_frac_empty = np.nanmean(frac_emptyISIs_per_fold)
    print("Mean accuracy: %f, mean frac empty ISIs: %f" %(mean_accuracy, mean_frac_empty))

    # Testing model on unseen data
    test_file = 'AE_238_2_AC.pickle'
    nwb_save_path = "./test_trials.nwb" 

    test_session, conditions_test = test_pickle_to_pynapple(
        pickle_path=test_file, 
        nwb_save_path=nwb_save_path, 
        iset_name="test trials",
        eset_name="lick events")
    print("Done making test Session object")

    conditions_test = np.asarray(conditions_test.values, dtype=str)

    log_ISIs_test = test_session.compute_log_ISIs("test trials")
    print("Done computing test log_ISIs")

    pred_conditions = n_decoder.predict_conditions(log_ISIs_test)
    print("Predicted conditions: ", pred_conditions)

    all_conditions_probs = n_decoder.predict_condition_probs(log_ISIs_test)
    print("Predicted probabilities for all conditions: ", all_conditions_probs)

    test_accuracy, test_frac_empty = n_decoder.calculate_accuracy(log_ISIs_test, conditions_test)
    print("Test accuracy: %f, frac empty ISIs: %f" %(test_accuracy, test_frac_empty))

    print("Saving and then loading NDecoder from pickle file")
    test_saving_loading_ndecoder(n_decoder=n_decoder, file_path="ndecoder.pickle")
    print("Loaded NDecoder's target PDF evaluated at 0:")
    print(n_decoder.model.conds['target'].pdf(0))


if __name__ == "__main__":
    test_workflow()


