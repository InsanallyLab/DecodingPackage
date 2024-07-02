from core.ndecoder import NDecoder
import numpy as np
from tests.io.test_io import test_saving_loading_ndecoder, test_saving_loading_log_ISIs, test_saving_loading_spikes, test_pickle_to_pynapple
    
def run_workflow():
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

    print("IO test: saving and loading spike train from npz file")
    test_saving_loading_spikes(
        session=train_session,
        iset_name="train trials",
        file_path="spike_train.npz",
        lock_point="start")

    print("IO test: saving and loading log ISIs from npz file")
    test_saving_loading_log_ISIs(
        session=train_session,
        iset_name="train trials",
        file_path="log_ISIs.npz",
        lock_point="start")

    log_ISIs_concat = np.concatenate(log_ISIs_train)
    bw_folds = 10    # from Insanally paper
    # TODO: grid_search is taking too long
    # kde_bw = Bandwidth.sklearn_grid_search_bw(log_ISIs_concat, bw_folds)
    kde_bw = 0.20
    print("KDE bandwidth: ", kde_bw)

    min_ISIs = 0 
    possible_conditions = ["target", "non-target"]
    n_decoder = NDecoder(bw=kde_bw, min_ISIs=min_ISIs, conditions=possible_conditions)
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

    print("IO test: saving and loading NDecoder from pickle file")
    test_saving_loading_ndecoder(n_decoder=n_decoder, file_path="ndecoder.pickle")


if __name__ == "__main__":
    run_workflow()