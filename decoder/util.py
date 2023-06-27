import numpy as np 

##from utility.py 
def generateDateString(meta):
    namemodifier = 'ERROR'
    if meta.task == 'CNO':
        namemodifier = 'CNO'
    elif meta.task in ['nonreversal','switch','reversal','second switch','second reversal']:
        namemodifier = str(meta.day_of_recording)
    elif meta.task in ['opto nonreversal','opto switch','opto reversal']:
        namemodifier = str(meta.day_of_recording)
    elif meta.task in ['tuning nonreversal','tuning switch','tuning reversal']:
        namemodifier = str(meta.day_of_recording)
    elif meta.task == 'thalamus tuning':
        namemodifier = str(meta.day_of_recording)

    return meta.animal + '_' + namemodifier + '_' + meta.region + '_' + str(meta.date).replace('/','-')


def getSpikeTimes(spikes,clust=np.nan,starttime=np.nan,endtime=np.nan,cachedtimes=None):
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
    if (not cachedtimes is None) and (not np.isnan(clust)):
        print('ERROR: Cannot split cached spike times according to unit. Please amend query')
        raise Exception

    #If we've not cached a search already, we need to reset our cache to the total times
    if cachedtimes is None:
        cachedtimes = spikes["times"]

    #Prune data by cluster. Remove spikes from all others
    if not np.isnan(clust):
        clustidx = np.equal(spikes["clusters"],clust)
        cachedtimes = cachedtimes[clustidx]

    #Remove spikes before starttime if applicable
    if not np.isnan(starttime):
        startidx = np.greater(cachedtimes,starttime)
        cachedtimes = cachedtimes[startidx]

    #Remove spikes after endtime if applicable 
    if not np.isnan(endtime):
        endidx = np.less(cachedtimes,endtime)
        cachedtimes = cachedtimes[endidx]

    return cachedtimes



