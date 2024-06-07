# Generating the traces you need: a conditional generative model for Process Mining data

Code and additional resources for paper "Generating the traces you need: a conditional generative model for Process Mining data".

## Additional material

Model weights, evaluation plots and additional material can be found in the folder `additional_material`.

## Installation

1. Create a new conda env: `conda create --name cvae-process-mining python=3.11` and activate it
2. Install required packages: `pip install -r requirements.txt`

## Run

1. Copy `config_example.py` to `config.py` and edit its content to configure training and evaluation
2. Run `python run_pipeline.py` to train the CVAE. A list of available command line arguments can be found in `arg_parser.py`.
3. Run `python evaluate.py` to perform evaluation

Training results can be found in folder `raytune`. You can also view results in Tensorboard. After installing tensorboardX (`pip install tensorboardX`) run the following command:
`tensorboard --logdir=<path_to_result_folder>`.

Datasets provided in the repository have already been preprocessed and splitted. However, if you want to do it yourself, we provide the scripts:

- `scripts/preprocess_log.py` in order to transform timestamps to relative timestamps to be fed to CVAE
- `scripts/split_log.py` in order to split the preprocessed log into train, val, test