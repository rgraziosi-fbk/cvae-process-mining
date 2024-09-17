import os
import sys
from math import ceil
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

def plot_activity_distribution_by_resource(original_log_path, generated_log_path, dataset_info=None, output_path='output', output_filename='activity-distribution-by-resource.png'):
  if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

  original_log = read_log(
    original_log_path,
    separator=dataset_info['CSV_SEPARATOR'],
    verbose=False
  )

  generated_log = read_log(
    generated_log_path,
    separator=dataset_info['CSV_SEPARATOR'],
    verbose=False
  )

  resources = dataset_info['RESOURCES']
  resources = sorted(resources)

  n_rows = ceil(len(resources) ** 0.5)
  fig, axs = plt.subplots(n_rows, n_rows, figsize=(len(resources), len(resources)))
  axs = axs.flatten()

  for res_i, resource in enumerate(resources):
    original_log_filtered = original_log[original_log[dataset_info['RESOURCE_KEY']] == resource]
    generated_log_filtered = generated_log[generated_log[dataset_info['RESOURCE_KEY']] == resource]

    original_log_activity_distribution = original_log_filtered[dataset_info['ACTIVITY_KEY']].value_counts(normalize=True)
    generated_log_activity_distribution = generated_log_filtered[dataset_info['ACTIVITY_KEY']].value_counts(normalize=True)

    original_log_activities = original_log_activity_distribution.keys().tolist()
    generated_log_activities = generated_log_activity_distribution.keys().tolist()

    # add missing activities
    for activity in original_log_activities:
      if activity not in generated_log_activities:
        generated_log_activity_distribution[activity] = 0

    for activity in generated_log_activities:
      if activity not in original_log_activities:
        original_log_activity_distribution[activity] = 0

    original_plus_generated_activities = list(set(original_log_activities + generated_log_activities))

    original_log_activity_distribution = original_log_activity_distribution.sort_index()
    generated_log_activity_distribution = generated_log_activity_distribution.sort_index()

    original_log_activity_distribution_list = original_log_activity_distribution.tolist()
    generated_log_activity_distribution_list = generated_log_activity_distribution.tolist()

    assert len(original_log_activity_distribution_list) == len(generated_log_activity_distribution_list)


    axs[res_i].bar(original_plus_generated_activities, original_log_activity_distribution_list, alpha=0.5, label='Original', color='blue')
    axs[res_i].bar(original_plus_generated_activities, generated_log_activity_distribution_list, alpha=0.5, label='Generated', color='red')
    axs[res_i].set_title(resource)
    axs[res_i].xaxis.set_tick_params(rotation=90)

  fig.suptitle('Event Duration Distributions', fontsize=16)
  fig.subplots_adjust(hspace=1.0, wspace=0.5)
  legend_elements = [Line2D([0], [0], color='blue', lw=4, label='Original'),
                   Line2D([0], [0], color='red', lw=4, label='Generated')]
  fig.legend(handles=legend_elements, loc='upper right')

  plt.savefig(os.path.join(output_path, output_filename))

  # reset plot settings to default and close figures
  plt.clf()
  plt.figure(figsize=plt.rcParams.get('figure.figsize'))
  plt.close(fig)