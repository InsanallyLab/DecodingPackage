
import numpy as np 

class SpikeTrain:
    def __init__(self, spikes, cluster):
        self.spikes = spikes  
        self.cluster = cluster 
        self.cached_times = None

    def get_spike_times(self, starttime=np.nan,endtime=np.nan):
        """
        set clust to control what neuron id to search for
        set starttime and endtime (in samples) to control time span
        if not set, starttime and endtime will each default to the start
        and end of the recording
        pass cached searches into cachedtimes to speed up sequential reads
        from the same neuron's spike times

        returns spike times relative to start of recording in samples
        """
        #Since we don't actually return any spike cluster identity, we cannot cache it
        #and thus cannot split any cached data according to cluster identity. This could
        #be amended but it is not currently intended behavior to allow this operation.
        if (not self.cached_times is None) and (not np.isnan(self.cluster)):
            print('ERROR: Cannot split cached spike times according to unit. Please amend query')
            raise Exception
        #If we've not cached a search already, we need to reset our cache to the total times
        if self.cached_times is None:
            self.cached_times = self.spikes.times
        else: 
            return self.cached_times

        #Prune data by cluster. Remove spikes from all others
        if not np.isnan(self.cluster):
            clustidx = np.equal(self.spikes.clusters,self.cluster)
            self.cached_times = self.cached_times[clustidx]

        #Remove spikes before starttime if applicable
        if not np.isnan(starttime):
            startidx = np.greater(self.cached_times,starttime)
            self.cached_times = self.cached_times[startidx]

        #Remove spikes after endtime if applicable
        if not np.isnan(endtime):
            endidx = np.less(self.cached_times,endtime)
            self.cached_times = self.cached_times[endidx]

        return self.cached_times

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