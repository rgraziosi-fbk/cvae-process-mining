import os
from ray import tune

from dataset import GenericDataset

# Dataset
# bpic2012_a, bpic2012_b, bpic2012_c, sepsis, traffic_fines
DATASET_NAME = 'sepsis'

# Used values in the paper: bpic2012_a|b = 100, sepsis = 50, traffic_fines = 20
if DATASET_NAME == 'sepsis':
  MAX_TRACE_LENGTH = 50
elif DATASET_NAME in ['bpic2012_a', 'bpic2012_b', 'bpic2012_c']:
  MAX_TRACE_LENGTH = 100
elif DATASET_NAME == 'traffic_fines':
  MAX_TRACE_LENGTH = 20

NUM_LABELS = 4 if DATASET_NAME == 'bpic2012_c' else 2

# Trace attributes to consider for each dataset
TRACE_ATTRIBUTES_BY_DATASET = {
  'bpic2012_a': [
    'AMOUNT_REQ',
  ],

  'bpic2012_b': [
    'AMOUNT_REQ',
  ],

  'bpic2012_c': [
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
  'NUM_LABELS': NUM_LABELS,
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
  'MODEL_PATH': '/Users/riccardo/Documents/pdi/topics/data-augmentation/RESULTS/ProcessScienceCollection/cvae/sepsis/training_output/best-models/best-model-epoch-4461.pt',
  'INPUT_PATH': os.path.abspath('input'),
  'OUTPUT_PATH': os.path.abspath('output'),
  
  'SHOULD_GENERATE': False,
  'GENERATION': {
    'NUM_GENERATIONS': 10,
    'LABELS': {
      'deviant': 154,
      'regular': 783,
    },
  },

  'SHOULD_USE_CVAE': True,
  'SHOULD_USE_LOG_3': True,
  'SHOULD_USE_LSTM_1': True,
  'SHOULD_USE_LSTM_2': True,
  'SHOULD_USE_TRANSFORMER_1': True,
  'SHOULD_USE_TRANSFORMER_2': True,
  'SHOULD_USE_PROCESSGAN_1': True,

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
  'LOG_DISTANCE_MEASURES_TO_COMPUTE': ['cfld', 'ngram_2', 'ngram_3', 'red', 'ctd'],
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
    # 'AMOUNT_REQ': [i for i in range(0, 100_000, 1000)], # bpic2012_a|b|c
    'Age': [i for i in range(30, 90, 5)], # sepsis
    # 'amount': [i for i in range(0, 200, 20)], # traffic_fines
  },

  # additional settings
  'SEED': 42,
}