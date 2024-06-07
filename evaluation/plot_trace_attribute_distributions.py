import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

def plot_trace_attribute_distributions(
  original_log_path,
  generated_log_path,
  trace_attributes,
  output_path='output',
  output_filename='trace-attribute-distributions.png',
  original_log_trace_key='case:concept:name',
  generated_log_trace_key='case:concept:name',
  condition=None
):
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  original_log = read_log(original_log_path)
  generated_log = read_log(generated_log_path)

  if condition:
    original_log = original_log[original_log['label'] == condition]
    generated_log = generated_log[generated_log['label'] == condition]

  # collapse rows with same 'case:concept:name' column into one
  original_log = original_log.groupby(original_log_trace_key).first().reset_index()
  generated_log = generated_log.groupby(generated_log_trace_key).first().reset_index()

  default_plot_size = plt.rcParams.get('figure.figsize')
  n_rows = len(trace_attributes)
  fig, axs = plt.subplots(n_rows, figsize=(default_plot_size[0]*n_rows, default_plot_size[1]), squeeze=False)
  axs = axs.flatten()

  for i, (attr_name, attr_bins) in enumerate(trace_attributes.items()):
    axs[i].hist(original_log[attr_name], bins=attr_bins, orientation='vertical', color='blue', alpha=0.5, label=f'Original', density=False)
    axs[i].hist(generated_log[attr_name], bins=attr_bins, orientation='vertical', color='red', alpha=0.5, label='Generated', density=False)

    axs[i].set_title(attr_name)

  fig.suptitle('Trace attribute distributions', fontsize=16)
  fig.subplots_adjust(hspace=0.5)
  legend_elements = [Line2D([0], [0], color='blue', lw=4, label='Original'),
                   Line2D([0], [0], color='red', lw=4, label='Generated')]
  fig.legend(handles=legend_elements, loc='upper right')

  plt.savefig(os.path.join(output_path, output_filename))

  # reset plot settings to default
  plt.clf()
  plt.figure(figsize=default_plot_size)
