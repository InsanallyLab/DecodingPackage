import numpy as np 
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KernelDensity 


class Bandwidth(): 
    # TODO: rewrite this function with something more general? 
    @staticmethod
    def getBWs_elife2019():
        #RNN
        return np.linspace(.005, 0.305, 10) 

    @staticmethod
    def sklearn_grid_search_bw(log_ISIs, folds=50):
        """
        Performs a grid search to determine the best bandwidth for Kernel Density Estimation.
        
        Parameters:
        - log_ISIs: ndarray
            The precomputed logISIs.
        - folds: int, default=50
            The number of cross-validation folds.
            
        Returns:
        - float
            The best bandwidth parameter found.
        """
        
        # Ensure that we don't try to cross validate more than there are ISIs
        folds = np.min([folds, len(log_ISIs)])

        log_ISIs = log_ISIs.reshape(-1, 1)  # Required to make GridSearchCV work

        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': Bandwidth.getBWs_elife2019()},
                            cv=folds, 
                            verbose=10, n_jobs=-1)
        grid.fit(log_ISIs)
        return grid.best_params_['bandwidth']
  

