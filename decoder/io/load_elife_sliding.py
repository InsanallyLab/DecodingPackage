from cmath import nan
from decoder.data.animal_info import ANIMALS
from decoder.data.elife_io import load_events_spikes_script
import numpy as np
from decoder.core.session import Session
from decoder.core.bandwidth import Bandwidth
from decoder.core.unique_interval_set import UniqueIntervalSet
from decoder.core.ndecoder import NDecoder
from scipy.stats import mannwhitneyu
import pynapple as nap 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

incorrect_format_files = ['AC_11262013']

AC_stim_accuracies = []
PFC_stim_accuracies = []
AC_stim_pval_s = []
PFC_stim_pval_s = []

AC_choice_accuracies = []
PFC_choice_accuracies = []
AC_choice_pval_s = []
PFC_choice_pval_s = []

s_stim_accuracies = []
s_choice_accuracies = []

for animal in ANIMALS:
    if animal in incorrect_format_files:
        continue

    animal_dict = ANIMALS[animal]

    if animal_dict['include']:
        print(animal)

        spike_files = animal_dict['spike_files']
        event_files = animal_dict['event_files']
        neurons = animal_dict['choice_neurons']
    
        (event_set, spike_set) = load_events_spikes_script(neuron_num=neurons, spike_files=spike_files, event_files=event_files)
        
        variable_maps = animal_dict['variables']
        stim_variables = ['T', 'F']
        action_variables = ['NPT', 'NPF']

        stimulus_conditions = []
        choice_conditions = []
        all_trial_starts = []
        all_nose_pokes = []

        for events, spikes_list, variable_map in zip(event_set, zip(*spike_set), variable_maps):
            for stim_variable, action_variable in zip(stim_variables, action_variables):
                trial_starts = events[variable_map[stim_variable]]
                try:
                    nose_poke_times = np.array(events[variable_map[action_variable]])
                except:
                    nose_poke_times = None

                all_trial_starts.extend(trial_starts)

                for i, trial_start in enumerate(trial_starts):
                    if stim_variable == 'T':
                        stimulus_conditions.append('target')
                    else:
                        stimulus_conditions.append('non-target')

                    # finding nosepoke time
                    if nose_poke_times is None:
                        all_nose_pokes.append(np.nan) 
                        choice_conditions.append('U')
                    else:
                        if i == len(trial_starts) - 1:
                            index = (nose_poke_times > trial_start)
                        else:
                            index = (nose_poke_times > trial_start)*(nose_poke_times < trial_starts[i+1])
                        if sum(index) == 0:
                            nose_poke_time = np.nan
                            choice_conditions.append('no-go')
                        else:
                            nose_poke_time = nose_poke_times[index][0]
                            choice_conditions.append('go')
                        all_nose_pokes.append(nose_poke_time)
        
        all_nose_pokes = np.array(all_nose_pokes)
        all_trial_starts = np.array(all_trial_starts)

        sort_index = np.argsort(all_trial_starts)
        all_trial_starts = all_trial_starts[sort_index]
        all_nose_pokes = all_nose_pokes[sort_index]
        stimulus_conditions = np.array(stimulus_conditions)
        stimulus_conditions = stimulus_conditions[sort_index]
        choice_conditions = np.array(choice_conditions)
        choice_conditions = choice_conditions[sort_index]
        
        iset_name = "%s trials" %animal
        trial_iset = UniqueIntervalSet(
            name=iset_name, 
            start=all_trial_starts, 
            end=all_nose_pokes, 
            start_padding=-0.5,
            # end_padding=0.5,
            fill_end_nans="mean", 
            remove_overlaps="last")

        if trial_iset.idx_removed is not None:
            stimulus_conditions = np.delete(stimulus_conditions, trial_iset.idx_removed)
            choice_conditions = np.delete(choice_conditions, trial_iset.idx_removed)

        event_sets = {}
        interval_sets = {iset_name : trial_iset}

        stimulus_t = [i for i in range(len(stimulus_conditions))]
        stimulus_tsd = nap.Tsd(t=stimulus_t, d=stimulus_conditions)

        choice_t = [i for i in range(len(choice_conditions))]
        choice_tsd = nap.Tsd(t=choice_t, d=choice_conditions)

        num_neurons = len(spike_set)

        for neuron_num in range(num_neurons):
            spike_train = nap.Ts(t=spike_set[neuron_num][0])
                        
            session = Session(spike_times=spike_train, interval_sets=interval_sets, event_sets=event_sets)

            windowed_log_ISIs, windows, final_spikes = session.compute_windowed_log_ISIs(iset_name=iset_name)

            log_ISIs = session.compute_log_ISIs(iset_name=iset_name)

            start_end_times = list(zip(trial_iset.start, trial_iset.end))
            start_end_times = np.array(start_end_times, dtype='object')

            first_window_log_ISIs = []
            for window_log_ISIs in windowed_log_ISIs:
                if len(window_log_ISIs) > 0:
                    first_window_log_ISIs.extend(window_log_ISIs[0])

            first_window_log_ISIs = np.array(first_window_log_ISIs)

            bw_folds = 10
            if len(first_window_log_ISIs) < 10:
                if animal[0] == 'A':
                    AC_stim_accuracies.append(nan)
                    AC_stim_pval_s.append(nan)
                    AC_choice_accuracies.append(nan)
                    AC_choice_pval_s.append(nan)
                else: 
                    PFC_stim_accuracies.append(nan)
                    PFC_stim_pval_s.append(nan)
                    PFC_choice_accuracies.append(nan)
                    PFC_choice_pval_s.append(nan)
                continue
            kde_bw = Bandwidth.sklearn_grid_search_bw(first_window_log_ISIs, bw_folds)
            print("KDE bandwidth: ", kde_bw)

            # Make stimulus decoder. 
            min_ISIs = 1
            possible_conditions = ["target", "non-target"]
            stimulus_decoder = NDecoder(bw=kde_bw, min_ISIs=min_ISIs, conditions=possible_conditions)
            s_stimulus_decoder = NDecoder(bw=kde_bw, min_ISIs=min_ISIs, conditions=possible_conditions)

            # Make choice decoder. 
            min_ISIs = 1
            possible_conditions = ["go", "no-go"]
            choice_decoder = NDecoder(bw=kde_bw, min_ISIs=min_ISIs, conditions=possible_conditions)
            s_choice_decoder = NDecoder(bw=kde_bw, min_ISIs=min_ISIs, conditions=possible_conditions)

            ''' 
            STIMULUS DECODER TRAINING
            ''' 
            reps = 124   # 124 in Insanally paper
            K_fold_num = 10  # 10 in Insanally paper
            accuracy_per_fold = []
            s_accuracy_per_fold = []
            frac_empty_ISIs_per_fold = []
            s_frac_empty_ISIs_per_fold = []

            conditions = np.asarray(stimulus_tsd.values, dtype=str)

            for rep in (range(int(reps/K_fold_num))):
                train_validate_pairs, train_val_idx = stimulus_decoder.generate_stratified_K_folds(
                    windowed_log_ISIs, 
                    conditions, 
                    K_fold_num, 
                    return_indices=True)

                for k, (train_data, validate_data) in enumerate(train_validate_pairs):
                    train_X, train_y = train_data

                    train_idx, val_idx = train_val_idx[k]

                    stimulus_decoder.fit_window(train_X, train_y, windows)

                    # Compute fold validation accuracy 
                    validate_X, validate_y = validate_data

                    # Validate should be unwindowed
                    validate_X = log_ISIs[val_idx]
                    validate_final_spikes = final_spikes[val_idx]
                    validate_start_end_times = start_end_times[val_idx]

                    accuracy, frac_empty = stimulus_decoder.calculate_window_accuracy(
                        validate_X, 
                        validate_y, 
                        validate_final_spikes)
                    accuracy_per_fold.append(accuracy)
                    frac_empty_ISIs_per_fold.append(frac_empty)

                    # Make synthetic model and compute synthetic fold validation accuracy
                    s_stimulus_decoder.fit_window(train_X, train_y, windows, synthetic = True)
                    
                    s_accuracy, s_frac_empty = s_stimulus_decoder.calculate_window_accuracy(
                        validate_X, 
                        validate_y, 
                        validate_final_spikes,
                        synthetic=True, 
                        start_end_times=validate_start_end_times)

                    s_accuracy_per_fold.append(s_accuracy)
                    s_frac_empty_ISIs_per_fold.append(s_frac_empty)

            print("Stimulus model fit for animal %s, neuron %d" %(animal, neuron_num))
            mean_accuracy = np.nanmean(accuracy_per_fold)
            mean_frac_empty = np.nanmean(frac_empty_ISIs_per_fold)
            mean_s_accuracy = np.nanmean(s_accuracy_per_fold)
            print("Mean accuracy: %f, mean frac empty ISIs: %f" %(mean_accuracy, mean_frac_empty))
            print("SYNTHETIC mean accuracy: %f"%mean_s_accuracy)

            pval_s = mannwhitneyu(accuracy_per_fold,s_accuracy_per_fold).pvalue
            print("IS PVAL LESS: ", pval_s < 0.05)
            print()

            s_stim_accuracies.append(mean_s_accuracy)

            if animal[0] == 'A':
                AC_stim_accuracies.append(mean_accuracy)
                AC_stim_pval_s.append(pval_s)
            else: 
                PFC_stim_accuracies.append(mean_accuracy)
                PFC_stim_pval_s.append(pval_s)

            ''' 
            CHOICE DECODER TRAINING
            ''' 
            reps = 124   # 124 in Insanally paper
            K_fold_num = 10  # 10 in Insanally paper
            accuracy_per_fold = []
            s_accuracy_per_fold = []
            frac_empty_ISIs_per_fold = []
            s_frac_empty_ISIs_per_fold = []

            conditions = np.asarray(choice_tsd.values, dtype=str)

            for rep in (range(int(reps/K_fold_num))):
                train_validate_pairs, train_val_idx = choice_decoder.generate_stratified_K_folds(
                    windowed_log_ISIs, 
                    conditions, 
                    K_fold_num, 
                    return_indices=True)

                for k, (train_data, validate_data) in enumerate(train_validate_pairs):
                    train_X, train_y = train_data

                    train_idx, val_idx = train_val_idx[k]

                    choice_decoder.fit_window(train_X, train_y, windows)

                    # Compute fold validation accuracy 
                    validate_X, validate_y = validate_data

                    # Validate should be unwindowed
                    validate_X = log_ISIs[val_idx]
                    validate_final_spikes = final_spikes[val_idx]
                    validate_start_end_times = start_end_times[val_idx]

                    accuracy, frac_empty = choice_decoder.calculate_window_accuracy(
                        validate_X, 
                        validate_y, 
                        validate_final_spikes)
                    accuracy_per_fold.append(accuracy)
                    frac_empty_ISIs_per_fold.append(frac_empty)

                    # Make synthetic model and compute synthetic fold validation accuracy
                    s_choice_decoder.fit_window(train_X, train_y, windows, synthetic = True)
                    
                    s_accuracy, s_frac_empty = s_choice_decoder.calculate_window_accuracy(
                        validate_X, 
                        validate_y, 
                        validate_final_spikes,
                        synthetic=True, 
                        start_end_times=validate_start_end_times)

                    s_accuracy_per_fold.append(s_accuracy)
                    s_frac_empty_ISIs_per_fold.append(s_frac_empty)

            print("Choice model fit for animal %s, neuron %d" %(animal, neuron_num))
            mean_accuracy = np.nanmean(accuracy_per_fold)
            mean_frac_empty = np.nanmean(frac_empty_ISIs_per_fold)
            mean_s_accuracy = np.nanmean(s_accuracy_per_fold)
            print("Mean accuracy: %f, mean frac empty ISIs: %f" %(mean_accuracy, mean_frac_empty))
            print("SYNTHETIC mean accuracy: %f"%mean_s_accuracy)

            pval_s = mannwhitneyu(accuracy_per_fold,s_accuracy_per_fold).pvalue
            print("IS PVAL LESS: ", pval_s < 0.05)
            print()

            s_choice_accuracies.append(mean_s_accuracy)

            if animal[0] == 'A':
                AC_choice_accuracies.append(mean_accuracy)
                AC_choice_pval_s.append(pval_s)
            else: 
                PFC_choice_accuracies.append(mean_accuracy)
                PFC_choice_pval_s.append(pval_s)

np.savez(
    "window_accuracies_with_synthetic", 
    AC_stim_accuracies=AC_stim_accuracies,
    PFC_stim_accuracies=PFC_stim_accuracies,
    AC_choice_accuracies=AC_choice_accuracies,
    PFC_choice_accuracies=PFC_choice_accuracies,
    s_stim_accuracies=s_stim_accuracies,
    s_choice_accuracies=s_choice_accuracies,
    AC_stim_pval_s=AC_stim_pval_s,
    PFC_stim_pval_s=PFC_stim_pval_s,
    AC_choice_pval_s=AC_choice_pval_s,
    PFC_choice_pval_s=PFC_choice_pval_s)

plt.scatter(s_stim_accuracies, s_choice_accuracies, 'k')
plt.xlabel("Stimulus decoding performance")
plt.ylabel("Choice decoding performance")
plt.title("Synthetic spike trains")
plt.show()

is_SS_AC_stim, acc_threshold_AC_stim = NDecoder.get_threshold_mask(AC_stim_accuracies, AC_stim_pval_s)
is_SS_PFC_stim, acc_threshold_PFC_stim = NDecoder.get_threshold_mask(PFC_stim_accuracies, PFC_stim_pval_s)
is_SS_AC_choice, acc_threshold_AC_choice = NDecoder.get_threshold_mask(AC_choice_accuracies, AC_choice_pval_s)
is_SS_PFC_choice, acc_threshold_PFC_choice = NDecoder.get_threshold_mask(PFC_choice_accuracies, PFC_choice_pval_s)

AC_stim_accuracies = np.array(AC_stim_accuracies)
PFC_stim_accuracies = np.array(PFC_stim_accuracies)
AC_stim_accuracies = list(AC_stim_accuracies[is_SS_AC_stim])
PFC_stim_accuracies = list(PFC_stim_accuracies[is_SS_PFC_stim])
print("Num AC stim neurons: ", len(AC_stim_accuracies))
print("Num PFC stim neurons: ", len(PFC_stim_accuracies))

AC_choice_accuracies = np.array(AC_choice_accuracies)
PFC_choice_accuracies = np.array(PFC_choice_accuracies)
AC_choice_accuracies = list(AC_choice_accuracies[is_SS_AC_choice])
PFC_choice_accuracies = list(PFC_choice_accuracies[is_SS_PFC_choice])
print("Num AC choice neurons: ", len(AC_choice_accuracies))
print("Num PFC choice neurons: ", len(PFC_choice_accuracies))

print("AC n = ", np.count_nonzero(np.logical_or(is_SS_AC_stim, is_SS_AC_choice)))
print("PFC n = ", np.count_nonzero(np.logical_or(is_SS_PFC_stim, is_SS_PFC_choice)))

max_len = max(len(AC_stim_accuracies), max(len(PFC_stim_accuracies), max(len(AC_choice_accuracies), len(PFC_choice_accuracies))))

if len(AC_stim_accuracies) < max_len:
    for i in range(max_len - len(AC_stim_accuracies)):
        AC_stim_accuracies.append(None)

if len(PFC_stim_accuracies) < max_len:
    for i in range(max_len - len(PFC_stim_accuracies)):
        PFC_stim_accuracies.append(None)

if len(AC_choice_accuracies) < max_len:
    for i in range(max_len - len(AC_choice_accuracies)):
        AC_choice_accuracies.append(None)

if len(PFC_choice_accuracies) < max_len:
    for i in range(max_len - len(PFC_choice_accuracies)):
        PFC_choice_accuracies.append(None)

# Plot mean validation accuracies
data = {'AC stimulus': AC_stim_accuracies, 
    'AC choice': AC_choice_accuracies,
    'PFC stimulus': PFC_stim_accuracies, 
    'PFC choice': PFC_choice_accuracies}

df = pd.DataFrame(data)

sns.swarmplot(data=df)
plt.ylabel('Decoding accuracy')
plt.show()








            
            


    


