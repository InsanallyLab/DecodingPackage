from core.ndecoder import NDecoder
from core.bandwidth import Bandwidth
import numpy as np
from tests.io.test_io import test_saving_loading_ndecoder, test_saving_loading_log_ISIs, test_pickle_to_pynapple
    
def run_workflow():
    '''
    Function demonstrating minimal working example.
    Uses Neuroconv, Pynapple to load pickle file data as Pynapple objects.
    Runs decoding algorithm on Pynapple objects, generates decoder, saves 
    decoder as pickle file.  
    '''
    data_file = 'AE_238_1_AC.pickle'
    nwb_save_path = "./trials.nwb" 

    session, conditions = test_pickle_to_pynapple(
        pickle_path=data_file, 
        nwb_save_path=nwb_save_path,
        iset_name="trials",
        eset_name="lick events")

    print("Done making Session object")

    log_ISIs = session.compute_log_ISIs(iset_name="trials", lock_point="start")
    print("Done computing log_ISIs")

    print("IO test: saving and loading log ISIs from npz file")
    test_saving_loading_log_ISIs(
        session=session,
        iset_name="trials",
        file_path="log_ISIs.npz",
        lock_point="start")

    log_ISIs_concat = np.concatenate(log_ISIs)
    # Using hard-coded bandwidth just to provide a quick working example
    kde_bw = 0.26336
    print("KDE bandwidth: ", kde_bw)

    min_ISIs = 0
    possible_conditions = ["target", "non-target"]
    n_decoder = NDecoder(bw=kde_bw, min_ISIs=min_ISIs, conditions=possible_conditions)
    print("NDecoder made")

    # Using very few reps and folds just to provide a quick working example
    reps = 2 
    K_fold_num = 2 
    accuracy_per_fold = []
    frac_empty_ISIs_per_fold = []

    conditions = np.asarray(conditions.values, dtype=str)

    for rep in (range(int(reps/K_fold_num))):
        train_validate_pairs = n_decoder.generate_stratified_K_folds(log_ISIs, conditions, K_fold_num)
        print("Generated stratified folds")

        for k, (train_data, validate_data) in enumerate(train_validate_pairs):
            train_X, train_y = train_data
            n_decoder.fit(train_X, train_y)
            print("Model fit on fold: %d, rep: %d" %(k, rep))

            # Compute fold validation accuracy 
            validate_X, validate_y = validate_data
            accuracy, frac_empty = n_decoder.calculate_accuracy(validate_X, validate_y)
            print("Calculated accuracy: %f, fraction empty ISIs: %f" %(accuracy, frac_empty))
            accuracy_per_fold.append(accuracy)
            frac_empty_ISIs_per_fold.append(frac_empty)

    print("Model fitting complete")
    mean_accuracy = np.nanmean(accuracy_per_fold)
    mean_frac_empty = np.nanmean(frac_empty_ISIs_per_fold)
    print("Mean accuracy: %f, mean frac empty ISIs: %f" %(mean_accuracy, mean_frac_empty))

    print("IO test: saving and loading decoder from pickle file")
    test_saving_loading_ndecoder(n_decoder=n_decoder, file_path="ndecoder.pickle")


if __name__ == "__main__":
    run_workflow()