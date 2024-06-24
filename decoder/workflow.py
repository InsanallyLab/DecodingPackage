from core.event_set import EventSet
from core.ndecoder import NDecoder
from core.session import Session
from core.bandwidth import Bandwidth
from core.unique_interval_set import UniqueIntervalSet
import pickle 
import numpy as np
import pynapple as nap 
from decoder.io.pickle_to_nwb import pickle_to_nwb

def test_loading_pickle(file_name, interval_set_name):
    '''
        Old function to load pickle file information as np arrays. 
        Does not use Pynapple functionality.
    '''
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()

    spike_times = data.spikes.times

    starts = data.trials.starts
    ends = data.trials.ends
    assert(len(starts) == len(ends))

    interval_set = UniqueIntervalSet(name=interval_set_name, start=starts, end=ends)
    interval_sets = [interval_set]

    lick_times = data.behavior.lick_times
    lick_labels = ["lick" for time in lick_times]
    possible_labels = ["lick"]
    event_set = EventSet(name="lick events", timestamps=lick_times, labels=lick_labels, possible_values=possible_labels)
    event_sets = [event_set]

    target_bools = data.trials.target
    trial_conditions = ["target" if target_bool == True else "non-target" for target_bool in target_bools]

    train_session = Session(spike_times=spike_times, interval_sets=interval_sets, event_sets=event_sets)

    return train_session, trial_conditions

def convert_units(nap_object, input_units : str = "s"):
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
            Raises ValueError if unrecognized time unit is passed in.
    """
    if isinstance(nap_object, nap.core.time_series.Ts):
        return nap.Ts(t=nap_object.t, time_units=input_units)

    if isinstance(nap_object, nap.core.time_series.Tsd):
        return nap.Tsd(t=nap_object.t, d=nap_object.d, time_units=input_units)
    
    if isinstance(nap_object, nap.core.interval_set.IntervalSet):
        return nap.IntervalSet(start=nap_object.start, end=nap_object.end, time_units=input_units)


def test_pickle_to_pynapple(pickle_path, nwb_save_path, input_time_units=None):
    '''
        Current function to test load pickle file information using NeuroConv 
        & Pynapple. 
    '''
    pickle_to_nwb(pickle_path=pickle_path, nwb_save_path=nwb_save_path)

    data = nap.load_file(nwb_save_path)

    interval_set = data["trials"] # IntervalSet
    spike_times = data["spikes"] # Ts
    lick_series = data["licks"] # Tsd
    trial_conditions = data["trial_conditions"] # Tsd
    
    # Allows user to change time units if needed
    if input_time_units is not None:
        interval_set = convert_units(interval_set, input_time_units)
        spike_times = convert_units(spike_times, input_time_units)
        lick_series = convert_units(lick_series, input_time_units)

    # TODO: Make this UniqueIntervalSet instead of IntervalSet
    interval_set.name = "train trials"
    interval_sets = [interval_set]

    # TODO: try lick_series as Tsd instead of EventSet
    lick_times = lick_series.t
    lick_labels = ["lick" for time in lick_times]
    possible_labels = ["lick"]
    event_set = EventSet(name="lick events", timestamps=lick_times, labels=lick_labels, possible_values=possible_labels)
    event_sets = [event_set]

    train_session = Session(spike_times=spike_times, interval_sets=interval_sets, event_sets=event_sets)

    return train_session, trial_conditions


'''
    Function demonstrating minimal working example
    Uses Neuroconv, Pynapple to load pickle file information as Pynapple objects
    Runs decoding algorithm on Pynapple objects, generates model, saves model 
    as pickle file.  
'''
def test_workflow():
    train_file = 'AE_238_1_AC.pickle'
    nwb_save_path = "./trials.nwb" 

    train_session, conditions_train = test_pickle_to_pynapple(
        pickle_path=train_file, 
        nwb_save_path=nwb_save_path)

    # train_session, conditions_train = test_loading_pickle(train_file, "train trials")
    print("Done making train Session object")

    # for all trials, so shape is (num_trials, num_ISIs)
    log_ISIs_train = train_session.compute_log_ISIs(iset_name="train trials", lock_point="start")
    print("Done computing train log_ISIs")

    train_session.save_log_ISIs(iset_name="train trials", filename="log_ISIs.npz", lock_point="start")

    # Test loading log_ISIs from local npz file
    # print("Load log ISIs test")
    # data = np.load("log_ISIs.npz", allow_pickle=True)
    # log_ISIs_train = data["locked_log_ISIs"]
    # data.close()

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

    # CHANGED: conditions_train is Tsd instead of np array
    conditions_train = np.asarray(conditions_train.values, dtype=str)

    for rep in (range(int(reps/K_fold_num))):
        train_validate_pairs = n_decoder.generate_stratified_K_folds(log_ISIs_train, conditions_train, K_fold_num)
        print("Generated stratified folds")

        for K, (train_data, validate_data) in enumerate(train_validate_pairs):
            train_X, train_y = train_data
            n_decoder.fit(train_X, train_y)
            print("Model fit on fold: %d, rep: %d" %(K, rep))

            # compute fold accuracy 
            validate_X, validate_y = validate_data
            accuracy, frac_empty = n_decoder.calculateAccuracy(validate_X, validate_y)
            print("Calculated accuracy for validation: %f, fraction empty ISIs: %f" %(accuracy, frac_empty))
            accuracy_per_fold.append(accuracy)
            frac_emptyISIs_per_fold.append(frac_empty)


    print("Model fitting complete")
    mean_accuracy = np.nanmean(accuracy_per_fold)
    mean_frac_empty = np.nanmean(frac_emptyISIs_per_fold)
    print("Mean accuracy: %f, mean frac empty ISIs: %f" %(mean_accuracy, mean_frac_empty))

    # Testing model on unseen data
    test_file = 'AE_238_2_AC.pickle'

    test_session, conditions_test = test_loading_pickle(test_file, "test trials")
    print("Done making test Session object")

    # for all trials, so shape is (num_trials, num_ISIs)
    log_ISIs_test = test_session.compute_log_ISIs("test trials")
    print("Done computing test log_ISIs")

    pred_conditions = n_decoder.predict_conditions(log_ISIs_test)
    print("Predicted conditions: ", pred_conditions)

    all_conditions_probs = n_decoder.predict_condition_probs(log_ISIs_test)
    print("Predicted probabilities for all conditions: ", all_conditions_probs)

    test_accuracy, test_frac_empty = n_decoder.calculateAccuracy(log_ISIs_test, conditions_test)
    print("Test accuracy: %f, frac empty ISIs: %f" %(test_accuracy, test_frac_empty))

    print("Saving NDecoder as pickle")
    n_decoder.save_as_pickle("ndecoder.pickle")

    # Test loading NDecoder from local pickle file
    # file = open("ndecoder.pickle", 'rb')
    # n_decoder = pickle.load(file)
    # file.close()
    # print("Loading NDecoder from pickle")
    # print(n_decoder.model.conds['target'].pdf(0))


test_workflow()


