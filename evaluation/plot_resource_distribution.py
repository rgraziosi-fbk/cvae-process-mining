import os
import sys
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log


def plot_resource_distribution(original_log_path, generated_log_path, dataset_info=None, output_path='output', output_filename='resource-distribution.png'):
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

  original_resource_distribution = original_log[dataset_info['RESOURCE_KEY']].value_counts(normalize=True)
  generated_resource_distribution = generated_log[dataset_info['RESOURCE_KEY']].value_counts(normalize=True)

  # add missing resources
  for resource in resources:
    if resource not in original_resource_distribution.keys().tolist():
      original_resource_distribution[resource] = 0

    if resource not in generated_resource_distribution.keys().tolist():
      generated_resource_distribution[resource] = 0

  original_resource_distribution = original_resource_distribution.sort_index()
  generated_resource_distribution = generated_resource_distribution.sort_index()


  original_resource_distribution_list = original_resource_distribution.tolist()
  generated_resource_distribution_list = generated_resource_distribution.tolist()

  assert len(resources) == len(original_resource_distribution_list) == len(generated_resource_distribution_list)

  # plot a single histogram with the two distributions
  fig, ax = plt.subplots()
  ax.bar(resources, original_resource_distribution_list, alpha=0.5, label='Original', color='blue')
  ax.bar(resources, generated_resource_distribution_list, alpha=0.5, label='Generated', color='red')
  ax.set_xlabel('Resource')
  ax.set_ylabel('Frequency')
  ax.legend()
  plt.title('Resource distribution')
  plt.xticks(rotation=0)
  plt.tight_layout()

  plt.savefig(os.path.join(output_path, output_filename))

  plt.clf()
  plt.figure(figsize=plt.rcParams.get('figure.figsize'))
  plt.close(fig)
