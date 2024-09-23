import os
import numpy as np
from matplotlib import pyplot as plt

def plot_boxplots(metrics, output_path='output', output_filename='results.png'):
  """
  Plot boxplots for the given metrics.

  Parameters:
    metrics (dict): A dictionary containing the metrics to plot. The keys represent the metric names, and the values are dictionaries with method names as keys (e.g. TEST1, GEN) and a list of metric values as values.
    output_path (str): The path where the output file will be saved. Default is 'output'.
    output_filename (str): The name of the output file. Default is 'results.png'.
  """
  default_figsize = plt.rcParams.get('figure.figsize')
  fig, axs = plt.subplots(1, len(metrics), figsize=(default_figsize[0]*len(metrics), default_figsize[1]))
  axs = np.array([axs]) if len(metrics) == 1 else axs

  for ax, (metric_name, metric_values) in zip(axs, metrics.items()):
    labels, data = metric_values.keys(), metric_values.values()

    boxplot = ax.boxplot(data, patch_artist=True, medianprops={'color': 'black', 'linewidth': 1})
    
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(metric_name)

    possible_colors = ['blue', 'green', 'red', 'purple', 'brown', 'cyan', 'orange', 'pink', 'gray', 'olive']
    
    for patch, color in zip(boxplot['boxes'], possible_colors[:len(metrics)]):
      patch.set_facecolor(color)

  plt.tight_layout()
  plt.savefig(os.path.join(output_path, output_filename))

  # reset plot settings to default and close figures
  plt.clf()
  plt.figure(figsize=default_figsize)
  plt.close(fig)