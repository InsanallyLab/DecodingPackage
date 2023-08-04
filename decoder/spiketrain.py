import numpy as np 
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KernelDensity 
from KDEpy import FFTKDE 
from scipy.interpolate import interp1d 

class SpikeTrain:
    def __init__(self, session_info):
        self.current_session = session_info 

    @staticmethod
    def getBWs_elife2019(self): 
        """ 
        method that returns the bandwidth used in the 2019 elife paper for bayesian decoding
        """
        return np.linspace(.005, 0.305, 11) 

    def get_spike_train_data(self, category, condition):
        # Retrieve the spike train data for the specified category and condition.
        # Use self.spike_train_data and self.session_info.

        return 

    def sklearn_grid_search_bw(self, loader,clust,trialsPerDayLoaded,interval,folds = 50):
        conditions =  self.current_session.getAllConditions(loader,clust,trialsPerDayLoaded=trialsPerDayLoaded)
        
        trialsToUse = conditions['all_trials'].trials
        trialsToUse = np.random.permutation(trialsToUse)
        trialsToUse = trialsToUse[0:int(len(trialsToUse)/2)]

        LogISIs,_ = self.current_session.getLogISIs(interval = interval,trials = trialsToUse)
        #Ensure that we don't try to cross validate more than there are ISIs
        folds = np.min([folds,len(LogISIs)])

        LogISIs = LogISIs.reshape(-1, 1)#Required to make GridSearchCV work
        #print(f"There are {len(LogISIs)} ISIs for bw selection")
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': self.getBWs_elife2019()},
                        cv=folds) # 20-fold cross-validation
        grid.fit(LogISIs)
        return grid.best_params_['bandwidth']

    @staticmethod
    def estimate_isi_distribution(log_isis, bandwidth):
        """
        Estimate the Inter-Spike Interval (ISI) Distribution using Kernel Density Estimation (KDE).

        Args:
            log_isis (numpy.ndarray): An array of log-transformed Inter-Spike Interval (ISI) values.
            bandwidth (float): The bandwidth parameter for the KDE.

        Returns:
            tuple: A tuple containing two functions:
                - A function that interpolates the estimated probability density function (PDF).
                - A function to generate an inverse Cumulative Distribution Function (CDF) for sampling.
        """
        x = np.linspace(-2, 6, 100)
        y = FFTKDE(bw=bandwidth, kernel='gaussian').fit(log_isis, weights=None).evaluate(x)

        # Use scipy to interpolate and evaluate on an arbitrary grid
        pdf_interpolator = interp1d(x, y, kind='linear', assume_sorted=True)

        # Generate an inverse CDF for sampling
        norm_y = np.cumsum(y) / np.sum(y)
        norm_y[0] = 0
        norm_y[-1] = 1
        cdf_inverse_interpolator = interp1d(norm_y, x, kind='linear', assume_sorted=True)

        return pdf_interpolator, cdf_inverse_interpolator

    @staticmethod
    def synthetic_spiketrain(model, trial_length=2500): 
        """ 
        generating synthetic spiketrains 
        """
        LogISIs = []
        ctime = 0

        while ctime <= trial_length:
            ISI = model.all.Inv_Likelihood(np.random.rand())
            ctime += 10 ** ISI
            if ctime <= trial_length:
                LogISIs.append(ISI)

        return np.array(LogISIs)
