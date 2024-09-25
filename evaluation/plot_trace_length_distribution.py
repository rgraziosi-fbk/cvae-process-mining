import os
import torch
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

def plot_trace_length_distribution(
  original_dataset_path,
  generated_log_path,
  dataset_info=None,
  output_path='output',
  output_filename='trace-length-distribution.png',
  generated_log_csv_separator=None,
  generated_log_trace_key=None,
):
  """
  Plot a histogram of trace lengths for original and generated datasets
  """

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  if generated_log_csv_separator is None:
    generated_log_csv_separator = dataset_info['CSV_SEPARATOR']

  if generated_log_trace_key is None:
    generated_log_trace_key = dataset_info['TRACE_KEY']

  # choose csv version of log
  generated_log_path = generated_log_path.replace('.xes', '.csv', 1)

  original_log = pd.read_csv(original_dataset_path, sep=dataset_info['CSV_SEPARATOR'])
  original_log_grouped_counts = original_log.groupby(dataset_info['TRACE_KEY']).size().reset_index(name='counts')
  original_log_lens = original_log_grouped_counts['counts'].tolist()
  original_log_lens = [v if v <= dataset_info['MAX_TRACE_LENGTH'] else dataset_info['MAX_TRACE_LENGTH'] for v in original_log_lens] # trim longer traces in original log

  generated_log = pd.read_csv(generated_log_path, sep=generated_log_csv_separator)
  generated_log_grouped_counts = generated_log.groupby(generated_log_trace_key).size().reset_index(name='counts')
  generated_log_lens = generated_log_grouped_counts['counts'].tolist()

  matplotlib.use('agg')
  plt.clf()

  bins = [i for i in range(dataset_info['MAX_TRACE_LENGTH'])]
  plt.hist(original_log_lens, bins=bins, label=f'Original', color='blue', alpha=0.5, density=False)
  plt.hist(generated_log_lens, bins=bins, label='Generated', color='red', alpha=0.5, density=False)

  plt.title('Trace length distribution')
  plt.xlabel('Trace length')
  plt.ylabel('Num of traces')
  plt.legend(loc='upper right')

  plt.savefig(os.path.join(output_path, output_filename))