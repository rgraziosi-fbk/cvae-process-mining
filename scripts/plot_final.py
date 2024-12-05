import copy
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CONFORMANCE_METRIC_KEY = 'conformance'
CFLD_METRIC_KEY = 'cfld'
NGRAM_2_METRIC_KEY = 'ngram_2'
CTD_METRIC_KEY = 'ctd'
RED_METRIC_KEY = 'red'
CWD_METRIC_KEY = 'cwd'

SEPSIS_LOG_KEY = 'sepsis'
BPIC2012_A_LOG_KEY = 'bpic2012_a'
BPIC2012_B_LOG_KEY = 'bpic2012_b'
TRAFFIC_FINES_LOG_KEY = 'traffic_fines'

LOG_METHOD_KEY = 'LOG_3'
CVAE_METHOD_KEY = 'CVAE'
CVAE_OLD_METHOD_KEY = 'OLD_CVAE'
LSTM_1_METHOD_KEY = 'LSTM_1'
LSTM_2_METHOD_KEY = 'LSTM_2'
PROCESSGAN_1_METHOD_KEY = 'PG_1'
PROCESSGAN_2_METHOD_KEY = 'PG_2'

FILTER_BY = 'full'
METRICS_PATH = '/Users/riccardo/Documents/pdi/topics/data-augmentation/cvae-process-mining/additional_material/process_science/metrics/'

COLORS = ['#c3e0f6', '#ea7693', '#d23359', '#73b459', '#f7c242', '#f7f7f7', '#d3d3d3', '#a9a9a9', '#7f7f7f', '#595959', '#262626']

# Mapping of metrics|logs|methods name

metric_key2name = {
  CONFORMANCE_METRIC_KEY: 'CONF',
  CFLD_METRIC_KEY: 'CFLD',
  NGRAM_2_METRIC_KEY: '2GD',
  CTD_METRIC_KEY: 'CTD',
  RED_METRIC_KEY: 'RED',
  CWD_METRIC_KEY: 'CWD',
}

log_key2name = {
  SEPSIS_LOG_KEY: 'Sepsis',
  BPIC2012_A_LOG_KEY: 'Bpic2012_1',
  BPIC2012_B_LOG_KEY: 'Bpic2012_2',
  TRAFFIC_FINES_LOG_KEY: 'Traffic_Fines',
}

method_key2name = {
  LOG_METHOD_KEY: 'train_log',
  CVAE_METHOD_KEY: 'cvae',
  CVAE_OLD_METHOD_KEY: 'old_cvae',
  LSTM_1_METHOD_KEY: 'lstm_1',
  LSTM_2_METHOD_KEY: 'lstm_2',
  PROCESSGAN_1_METHOD_KEY: 'processgan_1',
  PROCESSGAN_2_METHOD_KEY: 'processgan_2',
}

# Configuration

LOGS_TO_PLOT = [
  f'{SEPSIS_LOG_KEY}-{FILTER_BY}.json',
  # f'{BPIC2012_A_LOG_KEY}-{FILTER_BY}.json',
  f'{BPIC2012_B_LOG_KEY}-{FILTER_BY}.json',
  f'{TRAFFIC_FINES_LOG_KEY}-{FILTER_BY}.json',
]

METRICS_TO_PLOT = [
  # CONFORMANCE_METRIC_KEY,

  CFLD_METRIC_KEY,
  NGRAM_2_METRIC_KEY,

  # RED_METRIC_KEY,
  # CTD_METRIC_KEY,

  # CWD_METRIC_KEY,
]

METHODS_TO_PLOT = [
  LOG_METHOD_KEY,

  CVAE_METHOD_KEY,
  CVAE_OLD_METHOD_KEY,

  LSTM_1_METHOD_KEY,
  LSTM_2_METHOD_KEY,

  PROCESSGAN_1_METHOD_KEY,
  PROCESSGAN_2_METHOD_KEY,
]



# Plot desired metrics

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
fig, axs = plt.subplots(len(METRICS_TO_PLOT), len(LOGS_TO_PLOT), figsize=(default_figsize[0]*len(LOGS_TO_PLOT), default_figsize[1]*len(METRICS_TO_PLOT)))

for i, metric in enumerate(METRICS_TO_PLOT):
  for j, log in enumerate(logs_metrics):
    if len(METRICS_TO_PLOT) == 1:
      ax = axs[j]
    else:
      ax = axs[i, j]

    labels, data = logs_metrics[j][metric].keys(), logs_metrics[j][metric].values()

    boxplot = ax.boxplot(data, patch_artist=True, medianprops={'color': 'black', 'linewidth': 0.5})
    
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # write the log name
    if i == 0:
      log_key = LOGS_TO_PLOT[j].split('-')[0]
      ax.set_title(log_key2name[log_key], fontsize=16, fontweight='bold', pad=20)

    # write the metric name
    if j == 0:
      ax.set_ylabel(metric_key2name[metric], fontsize=16, fontweight='bold', labelpad=20)

    for patch, color in zip(boxplot['boxes'], COLORS[:len(METHODS_TO_PLOT)]):
      patch.set_facecolor(color)


# Create custom legend handles
legend_handles = []

for color, method in zip(COLORS, METHODS_TO_PLOT):
  legend_handles.append(mpatches.Patch(facecolor=color, label=method_key2name[method], edgecolor='black', linewidth=1))

# Add the custom legend to the figure
fig.legend(handles=legend_handles, loc='upper center', ncol=len(METHODS_TO_PLOT), fontsize='xx-large')

plt.tight_layout()

# Avoid overlapping of legend and plot
if len(METRICS_TO_PLOT) == 1:
  plt.subplots_adjust(top=0.75)
else:
  plt.subplots_adjust(top=0.85)

# Save the plot
plt.savefig(os.path.join(METRICS_PATH, 'plot.png'))

# reset plot settings to default and close figures
plt.clf()
plt.figure(figsize=default_figsize)
plt.close(fig)