import copy
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


FILTER_BY = 'full'
METRICS_PATH = '/Users/riccardo/Documents/pdi/topics/data-augmentation/dev/vae-gen-traces/scripts/metrics/'

LOGS_TO_PLOT = [
  f'sepsis_cases_1_custom-{FILTER_BY}.json',
  f'bpic2012_2-{FILTER_BY}.json',
  f'bpic2012_2_custom-{FILTER_BY}.json',
  f'traffic_fines_1_custom-{FILTER_BY}.json',
]
METRICS_TO_PLOT = ['red', 'ctd', 'ngram_2', 'conformance']
METHODS_TO_PLOT = ['LOG_4', 'LSTM_1', 'LSTM_2', 'VAE']

logs_metrics = []
# for each LOGS_TO_PLOT read json and append it to log_metrics
for metric_file_path in LOGS_TO_PLOT:
  with open(os.path.join(METRICS_PATH, metric_file_path)) as f:
    metric_file = json.load(f)
    metric_file_copy = copy.deepcopy(metric_file)

    # remove first-level keys that are not in METRICS_TO_PLOT
    for key in metric_file.keys():
      if key not in METRICS_TO_PLOT:
        metric_file_copy.pop(key)

    logs_metrics.append(metric_file_copy)


default_figsize = plt.rcParams.get('figure.figsize')
fig, axs = plt.subplots(len(LOGS_TO_PLOT), len(METRICS_TO_PLOT), figsize=(default_figsize[0]*len(METRICS_TO_PLOT), default_figsize[1]*len(LOGS_TO_PLOT)/1.75))

for i, log in enumerate(logs_metrics):
  for j, metric in enumerate(METRICS_TO_PLOT):
    ax = axs[i, j]

    labels, data = logs_metrics[i][metric].keys(), logs_metrics[i][metric].values()

    boxplot = ax.boxplot(data, patch_artist=True, medianprops={'color': 'black', 'linewidth': 1})
    
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # write the metric name
    if i == 0:
      metric_name = ''

      if metric == 'conformance': metric_name = 'CONF'
      elif metric == 'ngram_2': metric_name = '2GD'
      elif metric == 'red': metric_name = 'RED'
      elif metric == 'ctd': metric_name = 'CTD'

      ax.set_title(metric_name, fontsize=16, fontweight='bold', pad=20)

    # write the log name
    if j == 0:
      log_name = LOGS_TO_PLOT[i].split('-')[0]

      if log_name == 'sepsis_cases_1_custom': log_name = 'Sepsis'
      elif log_name == 'bpic2012_2': log_name = 'Bpic2012_A'
      elif log_name == 'bpic2012_2_custom': log_name = 'Bpic2012_B'
      elif log_name == 'traffic_fines_1_custom': log_name = 'Traffic_Fines'

      ax.set_ylabel(log_name, fontsize=16, fontweight='bold', labelpad=20)

    possible_colors = ['#c3e0f6', '#ea7693', '#d23359', '#73b459']
    
    for patch, color in zip(boxplot['boxes'], possible_colors[:len(METHODS_TO_PLOT)]):
      patch.set_facecolor(color)


# Create custom legend handles
legend_handles = []

for color, method in zip(possible_colors, METHODS_TO_PLOT):

  if method == 'LOG_4': method = 'train_log'
  elif method == 'LSTM_1': method = 'lstm1'
  elif method == 'LSTM_2': method = 'lstm2'
  elif method == 'VAE': method = 'cvae'

  legend_handles.append(mpatches.Patch(facecolor=color, label=method, edgecolor='black', linewidth=1))

# Add the custom legend to the figure
fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.65, 1), ncol=4, fontsize='xx-large')

# plt.tight_layout()
plt.savefig(os.path.join(METRICS_PATH, 'plot.png'))

# reset plot settings to default and close figures
plt.clf()
plt.figure(figsize=default_figsize)
plt.close(fig)