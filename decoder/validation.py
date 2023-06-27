import numpy as np 
from sklearn.model_selection import StratifiedKFold 

################################### Validation ################################

def Train_Test_Split(trials,frac_test = 0.1):
    """
    Splits a set of trials into test and train datasets
    trials: set of trials available in dataset
    frac_test: fraction of trails to use for test (0 for leave-one-out)
    returns (train_trials,test_trials)
    """
    
    N = len(trials)
    
    #Test set size. Must be at least one
    N_test = int(frac_test * N)
    if N_test < 1:
        N_test = 1
        
    #Train set size. Must also be at least one.
    N_train = N - N_test
    if N_train < 1:
        print('ERROR: No training data. Test fraction likely too high')
        raise Exception
        
    test_idxs = np.concatenate(( [False]*N_train , [True]*N_test ))
    test_idxs = np.random.permutation(test_idxs)
    train_idxs = np.logical_not(test_idxs)
    
    test_trials = trials[test_idxs]
    train_trials = trials[train_idxs]
    
    return (train_trials,test_trials)

def K_fold_strat(sessionfile,trials,K):
    trials = np.array(trials)
    
    X = np.ones(len(trials))
    y = np.ones(len(trials))
    for idx,trial in enumerate(trials):
        if sessionfile.trials.target[trial] and sessionfile.trials.go[trial]:
            y[idx] = 1
        elif sessionfile.trials.target[trial] and not sessionfile.trials.go[trial]:
            y[idx] = 2
        elif not sessionfile.trials.target[trial] and sessionfile.trials.go[trial]:
            y[idx] = 3
        elif not sessionfile.trials.target[trial] and not sessionfile.trials.go[trial]:
            y[idx] = 4
    
    train_test_pairs = []
    skf = StratifiedKFold(n_splits=K,shuffle=True)
    for splitX,splitY in skf.split(X, y):
        train_trials = trials[splitX]
        test_trials = trials[splitY]
        train_test_pairs.append((train_trials,test_trials))
    return train_test_pairs
