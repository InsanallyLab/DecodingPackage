import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity 
from numpy.typing import NDArray
from typing import Optional


class Bandwidth(): 
    @staticmethod
    def getBWs_elife():
        return np.linspace(.001, 1.00, 100)

    @staticmethod
    def sklearn_grid_search_bw(
        log_ISIs: NDArray, 
        folds: int = 10, 
        bandwidth_range: Optional[NDArray] = None):
        """
        Performs a grid search to determine the best bandwidth for Kernel Density Estimation.
        
        Parameters
        ----------
        log_ISIs : ndarray
            The precomputed log ISIs.
        folds : int
            The number of cross-validation folds. Defaults to 10.
        bandwidth_range : ndarray, optional
            All bandwidth candidates to evaluate. Defaults to the candidates 
            returned by getBWs_elife2019() if no argument is passed in.
            
        Returns:
            float: the best bandwidth parameter found.
        """
        
        # Ensure that we don't try to cross validate more than there are ISIs
        folds = np.min([folds, len(log_ISIs)])

        log_ISIs = log_ISIs.reshape(-1, 1)  # Required to make GridSearchCV work

        if bandwidth_range is None:
            bandwidth_range = Bandwidth.getBWs_elife()

        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidth_range},
                            cv=folds, n_jobs=-1)
        grid.fit(log_ISIs)
        return grid.best_params_['bandwidth']
  

