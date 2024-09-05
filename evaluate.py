import pickle
import torch

from config import evaluation_config, config, DATASET_INFO, MAX_TRACE_LENGTH
from utils import get_dataset_attributes_info
from model import VAE
from evaluate_utils import evaluate_generation

# Load model checkpoint (pickle or torch)
if evaluation_config['MODEL_PATH'].endswith('.pkl'):
  with open(evaluation_config['MODEL_PATH'], mode='rb') as f:
    checkpoint = pickle.load(f)
elif evaluation_config['MODEL_PATH'].endswith('.pth') or evaluation_config['MODEL_PATH'].endswith('.pt'):
  checkpoint = torch.load(evaluation_config['MODEL_PATH'], map_location='cpu')
else:
  raise ValueError('Model checkpoint must be either a pickle file or a torch file')

dataset_attributes_info = get_dataset_attributes_info(
  dataset_path=DATASET_INFO['FULL'],
  activity_key=DATASET_INFO['ACTIVITY_KEY'],
  trace_key=DATASET_INFO['TRACE_KEY'],
  resource_key=DATASET_INFO['RESOURCE_KEY'],
  trace_attributes=DATASET_INFO['TRACE_ATTRIBUTE_KEYS'],
)
max_trace_length = MAX_TRACE_LENGTH if MAX_TRACE_LENGTH else dataset_attributes_info['max_trace_length']

DATASET_INFO['MAX_TRACE_LENGTH'] = max_trace_length + 1
DATASET_INFO['ACTIVITIES'] = dataset_attributes_info['activities']
DATASET_INFO['NUM_ACTIVITIES'] = len(dataset_attributes_info['activities']) + 1
DATASET_INFO['RESOURCES'] = dataset_attributes_info['resources']
DATASET_INFO['NUM_RESOURCES'] = len(dataset_attributes_info['resources']) + 1
DATASET_INFO['TRACE_ATTRIBUTES'] = dataset_attributes_info['trace_attributes']

config['C_DIM'] = DATASET_INFO['NUM_LABELS']

# Load model
model = VAE(
  trace_attributes=DATASET_INFO['TRACE_ATTRIBUTES'],
  num_activities=DATASET_INFO['NUM_ACTIVITIES'],
  num_resources=DATASET_INFO['NUM_RESOURCES'],
  max_trace_length=DATASET_INFO['MAX_TRACE_LENGTH'],
  num_lstm_layers=config['NUM_LSTM_LAYERS'],
  attr_e_dim=config['ATTR_E_DIM'],
  act_e_dim=config['ACT_E_DIM'],
  res_e_dim=config['RES_E_DIM'],
  cf_dim=config['CF_DIM'],
  c_dim=config['C_DIM'],
  z_dim=config['Z_DIM'],
  dropout_p=config['DROPOUT_P'],
).to('cpu')

model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate model
evaluate_generation(
  model,
  should_generate=evaluation_config['SHOULD_GENERATE'],
  generation_config=evaluation_config['GENERATION'],
  dataset_info=DATASET_INFO,
  input_path=evaluation_config['INPUT_PATH'],
  output_path=evaluation_config['OUTPUT_PATH'],

  should_use_log_4=evaluation_config['SHOULD_USE_LOG_4'],
  should_use_lstm_1=evaluation_config['SHOULD_USE_LSTM_1'],
  should_use_lstm_2=evaluation_config['SHOULD_USE_LSTM_2'],

  should_skip_all_metrics_computation=evaluation_config['SHOULD_SKIP_ALL_METRICS_COMPUTATION'],
  should_plot_boxplots=evaluation_config['SHOULD_PLOT_BOXPLOTS'],

  # conformance checking
  should_compute_conformance=evaluation_config['SHOULD_COMPUTE_CONFORMANCE'],
  consider_vacuity=evaluation_config['CONFORMANCE_CONSIDER_VACUITY'],
  min_support=evaluation_config['CONFORMANCE_MIN_SUPPORT'],
  itemsets_support=evaluation_config['CONFORMANCE_ITEMSETS_SUPPORT'],
  max_declare_cardinality=evaluation_config['CONFORMANCE_MAX_DECLARE_CARDINALITY'],

  # log distance measures
  log_distance_measures_to_compute=evaluation_config['LOG_DISTANCE_MEASURES_TO_COMPUTE'],
  log_distance_measures_also_compute_filtered_by=evaluation_config['LOG_DISTANCE_MEASURES_ALSO_COMPUTE_FILTERED_BY'],

  # t-sne plot
  should_plot_tsne=evaluation_config['SHOULD_PLOT_TSNE'],
  tsne_max_og_0=evaluation_config['TSNE_MAX_OG_0'],
  tsne_max_og_1=evaluation_config['TSNE_MAX_OG_1'],
  tsne_max_gen_0=evaluation_config['TSNE_MAX_GEN_0'],
  tsne_max_gen_1=evaluation_config['TSNE_MAX_GEN_1'],

  # trace length distribution
  should_plot_trace_length_distribution=evaluation_config['SHOULD_PLOT_TRACE_LENGTH_DISTRIBUTION'],

  # variant statistics
  should_compute_variant_stats=evaluation_config['SHOULD_COMPUTE_VARIANT_STATS'],

  # event duration distribution
  should_plot_activity_duration_distributions=evaluation_config['SHOULD_PLOT_ACTIVITY_DURATION_DISTRIBUTIONS'],
  activity_duration_distributions_filter_by_label=evaluation_config['ACTIVITY_DURATION_DISTRIBUTIONS_FILTER_BY_LABEL'],

  # resources
  should_plot_resource_distribution=evaluation_config['SHOULD_PLOT_RESOURCE_DISTRIBUTION'],
  should_plot_activity_by_resource_distribution=evaluation_config['SHOULD_PLOT_ACTIVITY_BY_RESOURCE_DISTRIBUTION'],

  # trace attribute distributions
  should_plot_trace_attribute_distributions=evaluation_config['SHOULD_PLOT_TRACE_ATTRIBUTE_DISTRIBUTIONS'],
  trace_attributes=evaluation_config['TRACE_ATTRIBUTES'],

  # additional settings
  seed=evaluation_config['SEED'],
)
