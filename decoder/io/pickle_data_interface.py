from neuroconv.basedatainterface import BaseDataInterface
import pickle
from pynwb.epoch import TimeIntervals
from pynwb.base import TimeSeries
from pynwb.misc import AnnotationSeries
import numpy as np
from pathlib import Path
from typing import Optional
from pynwb import NWBFile
import json
from datetime import datetime 
from numpy.typing import ArrayLike

class PickleDataInterface(BaseDataInterface):
    """ Interface for spike train data stored in Insanally Lab pickle files. """

    def __init__(
        self,
        file_path: str,
        read_kwargs: Optional[dict] = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        file_path : str
            Path to local pickle file that needs to be converted to NWB.
        read_kwargs : dict, optional
        verbose : bool, default: True
        """
        read_kwargs = read_kwargs or dict()
        super().__init__(file_path=file_path)
        self.verbose = verbose

        '''
        _read_file() initializes the following fields with Pickle file data:
            self.session_start_time
            self.spike_train
            self.starts 
            self.ends 
            self.event_times 
            self.event_labels
            self.trial_conditions
        '''
        self._read_kwargs = read_kwargs
        self._read_file(file_path, **read_kwargs)

        # NWB objects that will be added to the generated NWB file
        self.time_intervals = None
        self.spike_time_series = None
        self.event_series = None
        self.condition_series = None
    

    def _read_file(self, file_path: str, **read_kwargs):
        '''
        Reads relevant data from the pickle file stored in file_path. Stores
        all data as np.ndarrays.

        Insanally Lab pickle file format:
            data = pickle.load(file)
            session_start_time = data.meta.date # str
            spike_train = data.spikes.times # ndarray
            starts = data.trials.starts # ndarray
            ends = data.trials.ends # ndarray
            event_times = data.behavior.lick_times # ndarray
        '''
        file = open(file_path, 'rb')
        data = pickle.load(file)
        file.close()

        self.session_start_time = datetime.strptime(data.meta.date, "%m/%d/%Y")

        self.spike_train = data.spikes.times

        self.starts = data.trials.starts
        self.ends = data.trials.ends 
        if len(self.starts) != len(self.ends):
            raise ValueError("Length of start times must match length of end times")

        self.event_times = data.behavior.lick_times
        self.event_labels = np.array(["lick" for time in self.event_times])

        target_bools = data.trials.target
        self.trial_conditions = np.array(["target" if target_bool == True 
                                            else "non-target" 
                                            for target_bool in target_bools])


    def get_metadata(self) -> dict:
        """ Fills in metadata for PickleDataInterface. """

        metadata = super().get_metadata()
        metadata["TimeIntervals"] = dict(
            trials=dict(
                table_name="trial times",
                table_description=f"trial start and end times generated from {self.source_data['file_path']}"
            )
        )
        metadata["AnnotationSeries"] = dict(
            name="spikes",
            description= "spike train from trials"
        )
        metadata["TimeSeries"] = dict(
            events= dict(
                name="lick times",
                description= "lick times from trials"
            ),
            trial_conditions=dict(
                name="trial conditions",
                description= "target/non-target condition for each trial"
            ) 
        )

        """ Adds session start time information to the NWB metadata. This is a 
        required metadata field for NWB files. """
        metadata["NWBFile"]["session_start_time"] = self.session_start_time

        return metadata

    def get_metadata_schema(self) -> dict:
        """Safely loads metadata schema from .json file."""

        file_path = Path('pickle_schema.json')
        assert file_path.is_file(), f"{file_path} is not a file."
        assert file_path.suffix in (".json"), f"{file_path} is not a valid json file."

        with open(file=file_path, mode="r") as fp:
            dictionary = json.load(fp=fp)
        return dictionary

    def convert_to_time_intervals(
        self,
        name: str,
        starts: ArrayLike,
        ends: ArrayLike
    ) -> TimeIntervals:
        """
        Creates a TimeIntervals object with trial start and end times.

        Parameters
        ----------
        name : str
            The name of the TimeIntervals object.
        starts : array-like
            The start times of each trial. 
        ends : array-like
            The end times of each trial.

        Returns
        -------
        TimeIntervals object. 
        Pynapple converts NWB TimeIntervals objects into IntervalSets when 
        loading NWB files.
        """
        time_intervals = TimeIntervals(name=name)

        for start, end in zip(starts, ends):
            time_intervals.add_interval(start_time=start, stop_time=end)

        return time_intervals
    
    def convert_to_annotation_series(
        self, 
        name: str,
        timestamps: ArrayLike
    ) -> AnnotationSeries:
        """
        Creates a AnnotationSeries object with one-dimensional timestamps.

        Parameters
        ----------
        name : str
            The name of the AnnotationSeries object.
        timestamps : array-like
            The timestamps to be stored. 

        Returns
        -------
        AnnotationSeries object. 
        Pynapple converts NWB AnnotationSeries objects into Ts objects when 
        loading NWB files.
        """
        return AnnotationSeries(name=name, data=[], timestamps=timestamps)

    def convert_to_time_series(
        self,
        name: str,
        labels: ArrayLike,
        timestamps: ArrayLike,
        label_units: str = "unknown",
    ) -> TimeSeries:
        """
        Creates a TimeSeries object with timestamps and the associated labels.

        Parameters
        ----------
        name : str
            The name of the AnnotationSeries object.
        labels : array-like
            The labels to be stored.
        timestamps : array-like
            The timestamps to be stored. 
        label_units : str, optional
            TimeSeries objects require the units of the labels to be passed in. 
            Defaults to "unknown". 

        Returns
        -------
        TimeSeries object. 
        Pynapple converts NWB TimeSeries objects into Tsd objects when 
        loading NWB files.
        """
        return TimeSeries(name=name, data=labels, timestamps=timestamps, unit=label_units)


    def add_to_nwbfile(
        self,
        nwbfile: NWBFile, 
        metadata: Optional[dict] = None,
    ) -> NWBFile:
        """
        Converts trial time intervals, spike train, event timestamps and labels,
        and trial condition data into the appropriate NWB objects. Runs the NWB 
        conversion for the instantiated data interface.

        Parameters
        ----------
        nwbfile : NWBFile
            An in-memory NWBFile object to write to the location.
        metadata : dict, optional
            Metadata dictionary with information used to create the NWBFile when
            one does not exist or overwrite=True.
        
        Note: even though metadata is not used in this function, it is a 
        required parameter as per BaseDataInterface's definition of 
        add_to_nwbfile. 
        """
        self.time_intervals = self.convert_to_time_intervals(
            name="trial_times",
            starts=self.starts, 
            ends=self.ends)
        nwbfile.add_time_intervals(self.time_intervals)

        self.spike_time_series = self.convert_to_annotation_series(
            name="spikes",
            timestamps=self.spike_train)

        nwbfile.add_acquisition(self.spike_time_series)

        self.event_series = self.convert_to_time_series(
            name="licks",
            labels=self.event_labels,
            timestamps=self.event_times
        )
        nwbfile.add_acquisition(self.event_series)

        self.condition_series = self.convert_to_time_series(
            name="trial_conditions",
            labels=self.trial_conditions,
            timestamps=[i for i in range(len(self.trial_conditions))]
        )
        nwbfile.add_acquisition(self.condition_series)

        return nwbfile