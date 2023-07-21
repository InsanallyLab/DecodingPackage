
import pandas as pd
import numpy as np 
import pynapple as nap 

def load_session(session_type, path = None, data_dict=None):
    """Load data from a session and return a BaseLoader object.

    Args:
        path (str, optional): The path to the session data. Default is None.
        session_type (str, optional): The type of session data to load. 
        Possible args are: 
         - Neurosuite\n
        - Phy\n
        - Minian\n
        - Inscopix-cnmfe\n
        - Matlab-cnmfe\n
        - Suite2p
        - Custom
        data_dict (dict, optional): A dictionary containing 'trials', 'meta', 'spikes', and optional 'trim' data. This data is non-optional for custom loader
                                    Default is None. 

    Returns:
        BaseLoader: An instance of BaseLoader containing the loaded data.

    Raises: 
        ValueError: if session_type = "custom" and data_dict = None 
    """
    session_type = session_type.lower()
    if session_type == "custom": 
        if data_dict == None: 
            raise ValueError("Please provide data for custom loader")
        trials = data_dict['trials']
        meta = data_dict['meta']
        spikes = data_dict['spikes']
        trim = data_dict.get('trim', None)
        return BaseLoader(trials, meta, spikes, trim)
    else:
        session = nap.load_session(path, session_type) 

        #TODO: figure out the different ways to extract trial and spike info from session object in pynapple
        return BaseLoader(session.trials, session.meta, session.spikes)

class Meta:
    def __init__(self, meta):
        self.animal = meta.animal
        self.fs = meta.fs
        self.day_of_recording = meta.day_of_recording
        self.day_of_training = meta.day_of_training
        self.date = meta.date
        self.length_in_trials = meta.length_in_trials
        self.region = meta.region
        self.task = meta.task

        if hasattr(meta, "first_reversal_trial"):
            self.first_reversal_trial = meta.first_reversal_trial
        else:
            self.first_reversal_trial = None

    def __str__(self):
        attributes = [
            f"Animal: {self.animal}",
            f"Sampling Frequency: {self.fs}",
            f"Day of Recording: {self.day_of_recording}",
            f"Day of Training: {self.day_of_training}",
            f"Date: {self.date}",
            f"Length in Trials: {self.length_in_trials}",
            f"Region: {self.region}",
            f"Task: {self.task}",
            f"First Reversal Trial: {self.first_reversal_trial}"
        ]
        return "\n".join(attributes)

class BaseLoader: 

    def __init__(self, trials, meta, spikes, trim = None): 
        self.trials = self.load_trials(trials) 
        self.meta = Meta(meta)
        self.spikes = self.load_spikes(spikes)
        self.trimmed_trials = trim

    def load_trials(self, trials): 
        trials_go = np.array(trials.go) 
        trials_starts = np.array(trials.starts) 
        trials_target = np.array(trials.target) 
        trials_response = np.array(trials.response) 
        trials_laser_stimulation = None

        if hasattr(trials, "laser_stimulation"): 
            trials_laser_stimulation = trials.laser_stimulation 

        trials_dict = {"go": trials_go, "starts": trials_starts, "target": trials_target, "response": trials_response, 
                       "laser stimulation": trials_laser_stimulation} 
        
        return pd.DataFrame(trials_dict)

    def load_spikes(self, spikes): 

        times = spikes.times
        clusters = spikes.clusters 

        spike_dict = { 
            "times" : times, 
            "clusters": clusters
        }

        return pd.DataFrame(spike_dict)
