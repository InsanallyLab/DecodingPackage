from datetime import datetime
from zoneinfo import ZoneInfo
import pynapple as nap 
from decoder.io.pickle_data_interface import PickleDataInterface

def neuroconv_to_nwb(interface, save_path):
    metadata = interface.get_metadata()
    print("Metadata extracted")
    print(metadata)

    # Add the time zone information to the conversion
    session_start_time = datetime(2024, 6, 17, 12, 30, 0, tzinfo=ZoneInfo("EST"))
    metadata["NWBFile"]["session_start_time"] = session_start_time

    nwbfile = interface.run_conversion(nwbfile_path=save_path, metadata=metadata)

def pickle_to_nwb(pickle_path, nwb_save_path):
    pickle_interface = PickleDataInterface(pickle_path)
    neuroconv_to_nwb(pickle_interface, nwb_save_path)