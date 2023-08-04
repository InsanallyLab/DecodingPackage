import numpy as np 
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KernelDensity 

################################### Bandwidth ################################# (REDUNDANT CODE)

class Bandwidth(): 
    @staticmethod
    def getBWs_elife2019(self):
        #RNN
        return np.linspace(.005, 0.305, 11) 

    @staticmethod
    def sklearn_grid_search_bw(isi, all_conditions, interval,folds = 50):
        conditions =  all_conditions 
        trialsToUse = conditions['all_trials'].trials
        trialsToUse = np.random.permutation(trialsToUse)
        trialsToUse = trialsToUse[0:int(len(trialsToUse)/2)]
        LogISIs = isi.get_log_isis(interval, trialsToUse)

        #Ensure that we don't try to cross validate more than there are ISIs
        folds = np.min([folds,len(LogISIs)])

        LogISIs = LogISIs.reshape(-1, 1)#Required to make GridSearchCV work
        #print(f"There are {len(LogISIs)} ISIs for bw selection")
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': Bandwidth.getBWs_elife2019()},
                        cv=folds) # 20-fold cross-validation
        grid.fit(LogISIs)
        return grid.best_params_['bandwidth']

  

