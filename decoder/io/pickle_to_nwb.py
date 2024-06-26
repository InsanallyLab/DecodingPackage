from decoder.io.pickle_data_interface import PickleDataInterface

def pickle_to_nwb(pickle_path : str, nwb_save_path : str):
    """
    Generates a NWB file from a pickle file using a custom Neuroconv Pickle 
    Data Interface. Saves the NWB file locally at nwb_save_path.

    Parameters
    ----------
    pickle_path : str
        The file path of the pickle file to convert into a NWB file. 
    nwb_save_path : str
        The file path to save the generated NWB file at.
    """
    pickle_interface = PickleDataInterface(pickle_path)
    pickle_interface.run_conversion(nwbfile_path=nwb_save_path)