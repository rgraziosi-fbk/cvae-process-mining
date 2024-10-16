import os
from ray import tune

from dataset import GenericDataset

# Dataset
# bpic2012_a, bpic2012_b, sepsis, traffic_fines
DATASET_NAME = 'sepsis_new'

# Used values in the paper: bpic2012_a|b = 100, sepsis = 50, traffic_fines = 20
MAX_TRACE_LENGTH = 50

# Trace attributes to consider for each dataset
TRACE_ATTRIBUTES_BY_DATASET = {
  'bpic2012_a': [
    'AMOUNT_REQ',
  ],

  'bpic2012_b': [
    'AMOUNT_REQ',
  ],

  'sepsis': [
    'Diagnose',
    'Age',
  ],

  'traffic_fines': [
    'vehicleClass',
    'amount',
  ],
}

DATASET_INFO = {
  'CLASS': GenericDataset,
  'TRAIN': os.path.abspath(os.path.join('datasets', DATASET_NAME, f'{DATASET_NAME}_TRAIN.csv')),
  'VAL': os.path.abspath(os.path.join('datasets', DATASET_NAME, f'{DATASET_NAME}_VAL.csv')),
  'TEST': os.path.abspath(os.path.join('datasets', DATASET_NAME, f'{DATASET_NAME}_TEST.csv')),
  'FULL': os.path.abspath(os.path.join('datasets', DATASET_NAME, f'{DATASET_NAME}.csv')),
  'CSV_SEPARATOR': ';',
  'NUM_LABELS': 2,
  'TRACE_ATTRIBUTE_KEYS': ['relative_timestamp_from_start'] + TRACE_ATTRIBUTES_BY_DATASET[DATASET_NAME],
  'ACTIVITY_KEY': 'Activity',
  'TIMESTAMP_KEY': 'relative_timestamp_from_previous_activity',
  'RESOURCE_KEY': 'Resource',
  'TRACE_KEY': 'Case ID',
  'LABEL_KEY': 'label',
}

# Training
MAX_NUM_EPOCHS = 20000
NUM_SAMPLES = 1
CHECKPOINT_EVERY = 200

# Training config
# You can pass multiple values per hyperparameter to perform hyperopt optimization with ray tune
config = {
  'NUM_KL_ANNEALING_CYCLES': MAX_NUM_EPOCHS // 2500,
  'EARLY_STOPPING_PATIENCE': 500,
  'EARLY_STOPPING_MIN_DELTA_PERC': 0.05,
  'EARLY_STOPPING_DEBUG': True,
  'IS_AUTOREGRESSIVE': True,
  'NUM_LSTM_LAYERS': 1,
  'ATTR_E_DIM': 5,
  'ACT_E_DIM': 5,
  'RES_E_DIM': 5,
  'CF_DIM': 200,
  'Z_DIM': 10,
  'DROPOUT_P': 0.05,
  'LR': 3e-4,
  'BATCH_SIZE': 256,
}

# Evaluation config
evaluation_config = {
  'MODEL_PATH': '/Users/riccardo/Documents/pdi/topics/data-augmentation/RESULTS/ProcessScienceCollection/current_esperimenti/sepsis/20k-epochs/best-models/best-model-epoch-4492.pt',
  'INPUT_PATH': os.path.abspath('input'),
  'OUTPUT_PATH': os.path.abspath('output'),
  
  'SHOULD_GENERATE': True,
  'GENERATION': {
    'NUM_GENERATIONS': 10,
    'LABELS': {
      'deviant': 23,
      'regular': 134,
    },
  },

  'SHOULD_USE_VAE': True,
  'SHOULD_USE_LOG_4': True,
  'SHOULD_USE_LSTM_1': True,
  'SHOULD_USE_LSTM_2': True,
  'SHOULD_USE_TRANSFORMER': True,

  # control every metric computation
  'SHOULD_SKIP_ALL_METRICS_COMPUTATION': False,
  'SHOULD_PLOT_BOXPLOTS': True,

  # conformance checking
  'SHOULD_COMPUTE_CONFORMANCE': True,
  'CONFORMANCE_CONSIDER_VACUITY': False,
  'CONFORMANCE_MIN_SUPPORT': 0.9,
  'CONFORMANCE_ITEMSETS_SUPPORT': 0.9,
  'CONFORMANCE_MAX_DECLARE_CARDINALITY': 2,

  # log distance measures
  'LOG_DISTANCE_MEASURES_TO_COMPUTE': ['cfld', 'ngram_2', 'red', 'ctd'],
  'LOG_DISTANCE_MEASURES_ALSO_COMPUTE_FILTERED_BY': [],

  # t-sne plot
  'SHOULD_PLOT_TSNE': False,
  'TSNE_MAX_OG_0': -1,
  'TSNE_MAX_OG_1': -1,
  'TSNE_MAX_GEN_0': -1,
  'TSNE_MAX_GEN_1': -1,

  # trace length distribution
  'SHOULD_PLOT_TRACE_LENGTH_DISTRIBUTION': True,

  # variant statistics
  'SHOULD_COMPUTE_VARIANT_STATS': True,

  # event duration distribution
  'SHOULD_PLOT_ACTIVITY_DURATION_DISTRIBUTIONS': True,
  'ACTIVITY_DURATION_DISTRIBUTIONS_FILTER_BY_LABEL': None,

  # resources
  'SHOULD_PLOT_RESOURCE_DISTRIBUTION': True,
  'SHOULD_PLOT_ACTIVITY_BY_RESOURCE_DISTRIBUTION': True,

  # trace attribute distributions
  'SHOULD_PLOT_TRACE_ATTRIBUTE_DISTRIBUTIONS': True,
  'TRACE_ATTRIBUTES': {
    # 'AMOUNT_REQ': [i for i in range(0, 100_000, 1000)], # bpic2012_a|b
    'Age': [i for i in range(30, 90, 5)], # sepsis
    # 'amount': [i for i in range(0, 200, 20)], # traffic_fines
  },

  # additional settings
  'SEED': 42,
}