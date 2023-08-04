import numpy as np 
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KernelDensity 

class SpikeTrain:
    def __init__(self, session_info):
        self.current_session = session_info 

    @staticmethod
    def getBWs_elife2019(self): 
        """ 
        method that returns the bandwidth used in the 2019 elife paper for bayesian decoding
        """
        return np.linspace(.005, 0.305, 11) 

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

    

    
