# BISI: the Bayesian InterSpike Interval Package

BISI is a comprehensive package for the trial-by-trial ISI-based Bayesian decoding algorithm 
presented in Insanally et al. 
To accommodate a wide variety of data formats, BISI's data loading and pre-processing 
pipeline makes use of Pynapple data structures. BISI is designed to decode trial conditions from any experimental dataset, regardless of the specific experimental set-up or data storage format. 

## Directory Structure

The repository is structured as follows:

- `decoder/`
  - `core/`: contains all the objects used in BISI's core functionality.
      - `unique_interval_set.py`: inherits Pynapple's IntervalSet object, which is 
   used to store trial start and end times. Adds additional capabilities for 
   preprocessing raw trial data. 
      - `session.py`: converts raw experimental data into processed log ISI data for each trial.
      - `ndecoder.py`: represents the Bayesian ISI decoder. 
      - `model.py`: stores all the necessary probability data for a particular trial condition.
      - `bandwidth.py`: used to find the optimal bandwidth for Kernel Density Estimation (KDE).
  - `io/`: contains BISI's data loading and data saving pipeline 
      - `load_elife_sliding.py`: runs the full BISI workflow on the dataset used for Insanally et al. (2019).
      - `pickle_to_nwb.py`: generates a NWB file from a pickle file using a custom Neuroconv data loader stored in `pickle_data_interface.py`.
      - `utils.py`: provides additional functionality such as converting time units for trial start and end times. 

For a deep dive into the methods, parameters, and return values, check the comments provided in the source files.

## Getting Started

1. **Clone the Repository**: Start by cloning the repository to your local machine.
   
   ```bash
   git clone [repository_link]
   ```

2. **Basic Usage**: You'll find a minimal working example of a simplified BISI workflow below, 
which loads an example dataset from a pickle file, runs the BISI algorithm using a 
fixed window rather than a sliding window, and then saves the generated Bayesian
ISI decoder as a pickle file. 

   ```bash
   cd path_to_repository/decoder
   python3 basic_workflow.py
   ```

## Acknowledgments

This package is built on the foundational research conducted by Insanally et al. The trial-by-trial spike timing Bayesian decoding algorithm has been detailed in [this paper](https://elifesciences.org/articles/42409#s4). 
This package also uses [Pynapple](https://github.com/pynapple-org/pynapple) data structures for its I/O layer, and provides functionality to convert pickle files to NWB files using [Neuroconv](https://github.com/catalystneuro/neuroconv).

## License
