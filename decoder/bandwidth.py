import numpy as np 
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KernelDensity 
from .utility import * 
from .analysis import * 

################################### Bandwidth ################################# (REDUNDANT CODE)

class Bandwidth(): 
    @staticmethod
    def getBWs_elife2019(self):
        #RNN
        return np.linspace(.005, 0.305, 11) 

    @staticmethod
    def getLogISIs(self, loader,clust,trials,interval): #This code is probably redundant. Could be replaced entirely by cacheLogISIs
        ISIs = []
        times = []
        for trial in trials:
            starttime,endtime = interval._ToTimestamp(loader.trials,trial)
            spiketimes =  getSpikeTimes(loader.spikes,clust=clust,starttime = starttime, endtime = endtime)
            spiketimes = spiketimes * 1000 / loader.meta.fs

            ISIs.append(np.diff(spiketimes))
            times.append(spiketimes[1:])

        ISIs = np.concatenate(ISIs)
        LogISIs = np.log10(ISIs)
        times = np.concatenate(times)
        return LogISIs, times

    @classmethod
    def sklearn_grid_search_bw(cls, loader,clust,trialsPerDayLoaded,interval,folds = 50):
        conditions =  getAllConditions(loader,clust,trialsPerDayLoaded=trialsPerDayLoaded)
        
        trialsToUse = conditions['all_trials'].trials
        trialsToUse = np.random.permutation(trialsToUse)
        trialsToUse = trialsToUse[0:int(len(trialsToUse)/2)]

        LogISIs,_ = cls.getLogISIs(loader,clust,trialsToUse,interval)
        #Ensure that we don't try to cross validate more than there are ISIs
        folds = np.min([folds,len(LogISIs)])

        LogISIs = LogISIs.reshape(-1, 1)#Required to make GridSearchCV work
        #print(f"There are {len(LogISIs)} ISIs for bw selection")
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': cls.getBWs_elife2019()},
                        cv=folds) # 20-fold cross-validation
        grid.fit(LogISIs)
        return grid.best_params_['bandwidth']

  

