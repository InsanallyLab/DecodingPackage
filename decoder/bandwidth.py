import numpy as np 
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KernelDensity 


class Bandwidth(): 
    @staticmethod
    def getBWs_elife2019():
        #RNN
        return np.linspace(.005, 0.305, 11) 

    @staticmethod
    def sklearn_grid_search_bw(log_isis, folds=50):
        """
        Performs a grid search to determine the best bandwidth for Kernel Density Estimation.
        
        Parameters:
        - log_isis: ndarray
            The precomputed logISIs.
        - folds: int, default=50
            The number of cross-validation folds.
            
        Returns:
        - float
            The best bandwidth parameter found.
        """
        
        # Ensure that we don't try to cross validate more than there are ISIs
        folds = np.min([folds, len(log_isis)])

        log_isis = log_isis.reshape(-1, 1)  # Required to make GridSearchCV work


        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': Bandwidth.getBWs_elife2019()},
                            cv=folds)
        grid.fit(log_isis)
        return grid.best_params_['bandwidth']
  

