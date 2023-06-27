
import pandas as pd
import numpy as np 

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

class LoadSession: 

    def __init__(self, sessionfile): 
        self.trials = self.load_trials(sessionfile.trials) 
        self.meta = Meta(sessionfile.meta)
        self.spikes = self.load_spikes(sessionfile.spikes)

        self.trimmed_trials = sessionfile.trim


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
