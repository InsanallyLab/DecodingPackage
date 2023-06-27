
import pandas as pd
import numpy as np 

class Loader: 

    def __init__(self, sessionfile): 

        self.trials = self.load_trials(sessionfile.trials) 
        self.meta = self.load_metadata(sessionfile.meta)
        self.spikes = self.load_spikes(sessionfile.spikes)


    def load_trials(self, trials): 
        
        trials_go = np.array(trials.go) 
        trials_starts = np.array(trials.starts) 
        trials_target = np.array(trials.target) 
        trials_response = np.array(trials.resposne) 
        trials_laser_stimulation = None

        if hasattr(trials, "laser_stimulation"): 
            trials_laser_stimulation = trials.laser_stimulation 

        trials_dict = {"Trials Go": trials_go, "Trials Starts": trials_starts, "Trials Target": trials_target, "Trials Response": trials_response, 
                       "Trials Laser Stimulation": trials_laser_stimulation} 
        
        return pd.DataFrame(trials_dict)
    
    def load_metadata(self, meta): 


    def load_spikes(self, spikes): 