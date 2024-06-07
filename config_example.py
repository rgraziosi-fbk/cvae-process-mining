import os
from ray import tune

from dataset import GenericDataset

# Dataset
DATASET_NAME = 'traffic_fines_1_custom'
MAX_TRACE_LENGTH = 20

# bpic2015_4.csv
# DATASET_INFO = {
#   'CLASS': GenericDataset,
#   'TRAIN': os.path.abspath(os.path.join('datasets', 'bpic2015_4', f'{DATASET_NAME}_TRAIN.csv')),
#   'VAL': os.path.abspath(os.path.join('datasets', 'bpic2015_4', f'{DATASET_NAME}_VAL.csv')),
#   'TEST': os.path.abspath(os.path.join('datasets', 'bpic2015_4', f'{DATASET_NAME}_TEST.csv')),
#   'FULL': os.path.abspath(os.path.join('datasets', 'bpic2015_4', f'{DATASET_NAME}.csv')),
#   'CSV_SEPARATOR': ';',
#   'NUM_LABELS': 2,
#   'TRACE_ATTRIBUTE_KEYS': [
#     'SUMleges',
#     'question',
#     'relative_timestamp_from_start',
#   ],
#   'ACTIVITY_KEY': 'Activity',
#   'TIMESTAMP_KEY': 'relative_timestamp_from_previous_activity',
#   'TRACE_KEY': 'Case ID',
#   'LABEL_KEY': 'label',
# }

# bpic2012_2.csv
# DATASET_INFO = {
#   'CLASS': GenericDataset,
#   'TRAIN': os.path.abspath(os.path.join('datasets', 'bpic2012_2', f'{DATASET_NAME}_TRAIN.csv')),
#   'VAL': os.path.abspath(os.path.join('datasets', 'bpic2012_2', f'{DATASET_NAME}_VAL.csv')),
#   'TEST': os.path.abspath(os.path.join('datasets', 'bpic2012_2', f'{DATASET_NAME}_TEST.csv')),
#   'FULL': os.path.abspath(os.path.join('datasets', 'bpic2012_2', f'{DATASET_NAME}.csv')),
#   'CSV_SEPARATOR': ';',
#   'NUM_LABELS': 2,
#   'TRACE_ATTRIBUTE_KEYS': [
#     'AMOUNT_REQ',
#     'relative_timestamp_from_start',
#   ],
#   'ACTIVITY_KEY': 'Activity',
#   'TIMESTAMP_KEY': 'relative_timestamp_from_previous_activity',
#   'TRACE_KEY': 'Case ID',
#   'LABEL_KEY': 'label',
# }

# sepsis_cases_1.csv
# DATASET_INFO = {
#   'CLASS': GenericDataset,
#   'TRAIN': os.path.abspath(os.path.join('datasets', 'sepsis_cases_1_custom', f'{DATASET_NAME}_TRAIN.csv')),
#   'VAL': os.path.abspath(os.path.join('datasets', 'sepsis_cases_1_custom', f'{DATASET_NAME}_VAL.csv')),
#   'TEST': os.path.abspath(os.path.join('datasets', 'sepsis_cases_1_custom', f'{DATASET_NAME}_TEST.csv')),
#   'FULL': os.path.abspath(os.path.join('datasets', 'sepsis_cases_1_custom', f'{DATASET_NAME}.csv')),
#   'CSV_SEPARATOR': ';',
#   'NUM_LABELS': 2,
#   'TRACE_ATTRIBUTE_KEYS': [
#     'Diagnose',
#     'Age',
#     'relative_timestamp_from_start',
#   ],
#   'ACTIVITY_KEY': 'Activity',
#   'TIMESTAMP_KEY': 'relative_timestamp_from_previous_activity',
#   'TRACE_KEY': 'Case ID',
#   'LABEL_KEY': 'label',
# }

# traffic_fines_1_custom.csv
DATASET_INFO = {
  'CLASS': GenericDataset,
  'TRAIN': os.path.abspath(os.path.join('datasets', 'traffic_fines_1_custom', f'{DATASET_NAME}_TRAIN.csv')),
  'VAL': os.path.abspath(os.path.join('datasets', 'traffic_fines_1_custom', f'{DATASET_NAME}_VAL.csv')),
  'TEST': os.path.abspath(os.path.join('datasets', 'traffic_fines_1_custom', f'{DATASET_NAME}_TEST_cropped.csv')),
  'FULL': os.path.abspath(os.path.join('datasets', 'traffic_fines_1_custom', f'{DATASET_NAME}.csv')),
  'CSV_SEPARATOR': ';',
  'NUM_LABELS': 2,
  'TRACE_ATTRIBUTE_KEYS': [
    'vehicleClass',
    'amount',
    'relative_timestamp_from_start',
  ],
  'ACTIVITY_KEY': 'Activity',
  'TIMESTAMP_KEY': 'relative_timestamp_from_previous_activity',
  'TRACE_KEY': 'Case ID',
  'LABEL_KEY': 'label',
}

# Training
MAX_NUM_EPOCHS = 20000
NUM_SAMPLES = 1

config = {
  'TMP_PATH': os.path.abspath('tmp'),
  'NUM_KL_ANNEALING_CYCLES': MAX_NUM_EPOCHS // 2500,
  'EARLY_STOPPING_PATIENCE': MAX_NUM_EPOCHS,
  'EARLY_STOPPING_MIN_DELTA': 1,
  'EARLY_STOPPING_DEBUG': True,
  'IS_AUTOREGRESSIVE': True,
  'NUM_LSTM_LAYERS': 1,
  'ATTR_E_DIM': 5,
  'ACT_E_DIM': 5,
  'CF_DIM': 200,
  'Z_DIM': 10,
  'DROPOUT_P': 0.05,
  'LR': 3e-4,
  'BATCH_SIZE': 256,
}

# Evaluation
evaluation_config = {
  'MODEL_PATH': '/Users/riccardo/Desktop/RESULTS/traffic_fines_1/weights/1249/checkpoint.pt',
  'INPUT_PATH': os.path.abspath('input'),
  'OUTPUT_PATH': os.path.abspath('output'),
  
  'SHOULD_GENERATE': False,
  'GENERATION': {
    'NUM_GENERATIONS': 10,
    'LABELS': {
      'deviant': 259,
      'regular': 6221,
    },
  },

  'SHOULD_USE_LSTM_1': True,
  'SHOULD_USE_LSTM_2': True,

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
  'LOG_DISTANCE_MEASURES_TO_COMPUTE': ['ngram_2', 'aed', 'ctd'],
  'LOG_DISTANCE_MEASURES_ALSO_COMPUTE_FILTERED_BY': ['deviant', 'regular'],

  # t-sne plot
  'SHOULD_PLOT_TSNE': False,
  'TSNE_MAX_OG_0': -1,
  'TSNE_MAX_OG_1': -1,
  'TSNE_MAX_GEN_0': -1,
  'TSNE_MAX_GEN_1': -1,

  # trace length distribution
  'SHOULD_PLOT_TRACE_LENGTH_DISTRIBUTION': False,

  # variant statistics
  'SHOULD_COMPUTE_VARIANT_STATS': True,

  # event duration distribution
  'SHOULD_PLOT_ACTIVITY_DURATION_DISTRIBUTIONS': False,
  'ACTIVITY_DURATION_DISTRIBUTIONS_FILTER_BY_LABEL': None,

  # trace attribute distributions
  'SHOULD_PLOT_TRACE_ATTRIBUTE_DISTRIBUTIONS': False,
  'TRACE_ATTRIBUTES': {
    # 'Age': [i for i in range(30, 90, 5)], # sepsis_cases_1
    # 'AMOUNT_REQ': [i for i in range(0, 100_000, 1000)], # bpic2012_2
    # 'SUMleges': [i for i in range(-1000, 20_000, 1000)], # bpic2015_4
    'amount': [i for i in range(0, 200, 20)], # traffic_fines_1
  },

  # additional settings
  'SEED': 42,
}