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

