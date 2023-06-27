
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

        meta_first_reversal_trial = None 
        animal = meta.animal 
        fs = meta.fs 
        dor = meta.day_of_recording 
        dot = meta.day_of_training 
        date = meta.date 
        length = meta.length_in_trials 
        region = meta.region 
        task = meta.task 

        if hasattr(meta, "first_reversal_trial"): 
            meta_first_reversal_trial = meta.first_reversal_trial 

        meta_dict = { 
            "First Reversal Trial" : meta_first_reversal_trial, 
            "Animal" : animal, 
            "FS" : fs, 
            "Date of Recording": dor, 
            "Day of Training": dot, 
            "Date" : date, 
            "Length In Trials": length, 
            "Region": region, 
            "Task" : task 
        }

        return pd.DataFrame(meta_dict)


    def load_spikes(self, spikes): 

        times = spikes.times
        clusters = spikes.clusters 

        spike_dict = { 
            "Times" : times, 
            "Clusters": clusters
        }

        return pd.DataFrame(spike_dict)
