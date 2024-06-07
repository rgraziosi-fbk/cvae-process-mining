# Generating the traces you need: a conditional generative model for Process Mining data

Code and additional resources for paper "Generating the traces you need: a conditional generative model for Process Mining data".

## Installation

1. Create a new conda env: `conda create --name vae-gen-traces python=3.11` and activate it
2. Install required packages: `pip install -r requirements.txt`
3. Run the training pipeline: `python run_pipeline.py`

Run results can be found in folder `results`. You can also view results in Tensorboard. After installing tensorboardX (`pip install tensorboardX`) run the following command:
`tensorboard --logdir=<path_to_result_folder>`

## Run

1. Run `scripts/preprocess_log.py` in order to transform timestamps to relative timestamps
2. Run `scripts/split_log.py` in order to split the preprocessed log into train, val, test
3. Rename `config_example.py` to `config.py` and edit its content to configure training and evaluation configs
4. Run `run_pipeline.py` to train the model on the train set
5. Run `evaluate.py` to perform evaluation of generated logs
