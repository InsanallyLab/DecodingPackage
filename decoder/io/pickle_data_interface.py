from curses import meta
from neuroconv.basedatainterface import BaseDataInterface
import pickle
from pynwb.epoch import TimeIntervals
from pynwb.base import TimeSeries
from pynwb.misc import AnnotationSeries
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from pynwb import NWBFile
import json

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
        read_kwargs : dict, optional
        verbose : bool, default: True
        """
        read_kwargs = read_kwargs or dict()
        super().__init__(file_path=file_path)
        self.verbose = verbose

        # Will initialize the following:
        # self.spike_train
        # self.starts
        # self.ends
        # self.event_times
        # self.event_labels
        # self.trial_conditions
        self._read_kwargs = read_kwargs
        self._read_file(file_path, **read_kwargs)

        # To add to nwb file as nwb objects 
        self.time_intervals = None
        self.spike_time_series = None
        self.event_series = None
        self.condition_series = None
    

    def _read_file(self, file_path: str, **read_kwargs):
        '''
        Pickle file format:

            data = pickle.load(file)
            spike_train = data.spikes.times # ndarray
            starts = data.trials.starts # ndarray
            ends = data.trials.ends # ndarray

            incorporate later if needed?:
            lick_times = data.behavior.lick_times # ndarray
        '''
        file = open(file_path, 'rb')
        data = pickle.load(file)
        file.close()

        self.spike_train = data.spikes.times # ndarray

        self.starts = data.trials.starts # ndarray
        self.ends = data.trials.ends # ndarray
        if len(self.starts) != len(self.ends):
            raise ValueError("Length of start times must match length of end times")

        self.event_times = data.behavior.lick_times
        self.event_labels = ["lick" for time in self.event_times]

        target_bools = data.trials.target
        self.trial_conditions = ["target" if target_bool == True else "non-target" for target_bool in target_bools]


    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata["TimeIntervals"] = dict(
            trials=dict(
                table_name="trial times",
                table_description=f"trial start and end times generated from {self.source_data['file_path']}",
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
        return metadata

    def get_metadata_schema(self) -> dict:
        """Safely load metadata from .json file."""
        file_path = Path('pickle_schema.json')
        assert file_path.is_file(), f"{file_path} is not a file."
        assert file_path.suffix in (".json"), f"{file_path} is not a valid json file."

        with open(file=file_path, mode="r") as fp:
            dictionary = json.load(fp=fp)
        return dictionary


    '''
    Pynapple converts TimeIntervals into IntervalSets when loading nwb files
    '''
    def convert_to_time_intervals(
        self,
        starts,
        ends, 
        time_intervals_name: str,
        time_intervals_description: str
    ) -> TimeIntervals:
        """
        Create a TimeIntervals object with trial start and end times.

        Parameters
        ----------
        table_name : str, optional
            The name of the TimeIntervals object.

        table_description : str, optional
            The description of the TimeIntervals object.

        Returns
        -------
        TimeIntervals

        """
        time_intervals = TimeIntervals(name=time_intervals_name, 
                                       description=time_intervals_description)

        for start, end in zip(starts, ends):
            time_intervals.add_interval(start_time=start, stop_time=end)

        return time_intervals
    

    '''
    Pynapple converts AnnotationSeries into Ts when loading nwb files
    '''
    def convert_to_annotation_series(
        self, 
        name : str,
        data,
        timestamps
    ) -> AnnotationSeries:
        return AnnotationSeries(name=name, data=data, timestamps=timestamps)

    '''
    Pynapple converts 1D TimeSeries into Tsd when loading nwb files
    '''
    def convert_to_time_series(
        self,
        name : str,
        labels,
        timestamps,
        label_units : str = "unknown",
    ) -> TimeSeries:
        return TimeSeries(name=name, data=labels, timestamps=timestamps, unit=label_units)


    def add_to_nwbfile(
        self,
        nwbfile: NWBFile, 
        metadata: Optional[dict] = None,
    ) -> NWBFile:
        """
        Run the NWB conversion for the instantiated data interface.

        Parameters
        ----------
        nwbfile : NWBFile
            An in-memory NWBFile object to write to the location.
        metadata : dict, optional
            Metadata dictionary with information used to create the NWBFile when one does not exist or overwrite=True.
        
        Note: even though metadata is not used in this function, it is a required parameter as per BaseDataInterface's 
        definition of add_to_nwbfile. 
        """
        self.time_intervals = self.convert_to_time_intervals(
                                starts=self.starts, 
                                ends=self.ends,
                                time_intervals_name="trials",
                                time_intervals_description="trial start and end times")
        nwbfile.add_time_intervals(self.time_intervals)

        self.spike_time_series = self.convert_to_annotation_series(
            name="spikes",
            data=[],
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
