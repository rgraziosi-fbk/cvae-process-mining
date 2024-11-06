# Generating the Traces You Need: A Conditional Generative Model for Process Mining Data

Code and additional resources for paper "Generating the Traces You Need: A Conditional Generative Model for Process Mining Data".

## Paper

[Link to arXiv](https://arxiv.org/abs/2411.02131).

## Additional material

Model weights, evaluation plots and additional material can be found in the folder `additional_material`.

## Install

1. Create a new conda env: `conda create --name cvae-process-mining python=3.11` and activate it
2. Install required packages: `pip install -r requirements.txt`

Note: if package installation fails with `ERROR: Failed building wheel for cchardet`, try to run `pip install cython` and then install requirements again.

## Run

1. Copy `config_example.py` to `config.py` and edit its content to configure training and evaluation
2. Run `python run_pipeline.py` to train the CVAE. A list of available command line arguments can be found in `arg_parser.py`.
3. Run `python evaluate.py` to perform evaluation

Training results can be found in folder `raytune`. You can also view results in Tensorboard. After installing tensorboardX (`pip install tensorboardX`) run the following command:
`tensorboard --logdir=<path_to_result_folder>`.

Datasets provided in the repository have already been preprocessed and splitted. However, if you want to do it yourself, we provide the scripts:

- `scripts/preprocess_log.py` in order to transform timestamps to relative timestamps to be fed to CVAE
- `scripts/split_log.py` in order to split the preprocessed log into train, val, test

## How to cite

R. Graziosi et al., "Generating the Traces You Need: A Conditional Generative Model for Process Mining Data," 2024 6th International Conference on Process Mining (ICPM), Kgs. Lyngby, Denmark, 2024, pp. 25-32, doi: 10.1109/ICPM63005.2024.10680621. keywords: {Process mining;Measurement;Deep learning;Adaptation models;Analytical models;Process control;Data models;Process Mining;Deep Learning;Generative AI;Conditional models},


```
@INPROCEEDINGS{10680621,
  author={Graziosi, Riccardo and Ronzani, Massimiliano and Buliga, Andrei and Di Francescomarino, Chiara and Folino, Francesco and Ghidini, Chiara and Meneghello, Francesca and Pontieri, Luigi},
  booktitle={2024 6th International Conference on Process Mining (ICPM)}, 
  title={Generating the Traces You Need: A Conditional Generative Model for Process Mining Data}, 
  year={2024},
  volume={},
  number={},
  pages={25-32},
  keywords={Process mining;Measurement;Deep learning;Adaptation models;Analytical models;Process control;Data models;Process Mining;Deep Learning;Generative AI;Conditional models},
  doi={10.1109/ICPM63005.2024.10680621}}
```