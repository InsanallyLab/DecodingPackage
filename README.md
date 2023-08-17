# NDecode: Neural Decoding Package

NDecode is a comprehensive solution for the trial-by-trial Spike Timing Bayesian Decoding Algorithm. The project derives inspiration and its algorithmic base from the paper by Insanally et al., which can be accessed [here](https://elifesciences.org/articles/42409#s4).

## Directory Structure

The repository is structured as follows:

- `decoder/`: Main directory containing all the essential files and sub-modules.
  - `core/`: Core functionalities and implementations of the NDecoder class.
  - `io/`: Input and output operations related to the decoder.
  - `main.py`: The main script where the `NDecoder` class is instantiated and utilized.

## Getting Started

1. **Clone the Repository**: Start by cloning the repository to your local machine.
   
   ```bash
   git clone [repository_link]
   ```

2. **Navigate to the Directory**:

   ```bash
   cd path_to_repository/decoder
   ```
## Detailed Documentation

For a deep dive into the methods, parameters, and return values, check the comments and docstrings provided in the source files.

- Core functionalities: `decoder/core`
- IO Operations: `decoder/io`
- Main implementation and usage examples: `decoder/main.py`

## Contributing

We welcome contributors! If you find any bugs or wish to add a feature, please open an issue or submit a pull request.

## Acknowledgments

This project is built on the foundational research conducted by Insanally et al. The trial-by-trial Spike Timing Bayesian Decoding Algorithm, which is the core of this implementation, has been detailed in [this paper](https://elifesciences.org/articles/42409#s4) by Insanally et al.

## License
