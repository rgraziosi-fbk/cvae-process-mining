import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import ceil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

def plot_activity_duration_distributions(
  log_paths,
  dataset_info,
  output_path='output',
  output_filename='activity-duration-distributions.png',
  activity_duration_distributions_filter_by_label=None,
  case_col_name=['case:concept:name', 'case:concept:name'],
  timestamp_col_name=['time:timestamp', 'time:timestamp'],
  activity_col_name=['concept:name', 'concept:name']
):
  """
  logs is supposed to be a list of 2 logs: the first one is the original log, the second one is the generated log
  """
  assert len(log_paths) == 2

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  logs = [read_log(log_path, separator=dataset_info['CSV_SEPARATOR'], verbose=False) for log_path in log_paths]

  if activity_duration_distributions_filter_by_label:
    for i, log in enumerate(logs):
      logs[i] = log[log['label'] == activity_duration_distributions_filter_by_label]

  activities = set()

  for log_i, log in enumerate(logs):
    # add column 'duration' to the log
    log['duration'] = 0.0

    # compute durations
    traces = log[case_col_name[log_i]].unique().tolist()
    for trace in traces:
      trace_df = log[log[case_col_name[log_i]] == trace]
      for i in range(len(trace_df) - 1):
        mask = log[case_col_name[log_i]] == trace
        index = log.loc[mask].index[i]
        log.loc[index, 'duration'] = (trace_df.iloc[i + 1][timestamp_col_name[log_i]] - trace_df.iloc[i][timestamp_col_name[log_i]]).total_seconds() / 60.0
      
    activities.update(log[activity_col_name[log_i]].unique().tolist())

  activities = sorted(activities)

  n_rows = ceil(len(activities) ** 0.5)
  fig, axs = plt.subplots(n_rows, n_rows, figsize=(len(activities), len(activities)))
  axs = axs.flatten()

  # for each activity, get all the durations and plot an histogram
  for act_i, activity in enumerate(activities):
    # compute plot_range for both logs (to have the same scale)
    # i.e. take the min and max of the durations of the two logs
    plot_range = [0, 0]
    for log_i, log in enumerate(logs):
      durations = log[log[activity_col_name[log_i]] == activity]['duration']
      plot_range[0] = min(plot_range[0], durations.min())
      plot_range[1] = max(plot_range[1], durations.max())

    for log_i, log in enumerate(logs):
      durations = log[log[activity_col_name[log_i]] == activity]['duration']

      plot_color = 'blue' if log_i == 0 else 'red'
      plot_label = 'Original' if log_i == 0 else 'Generated'

      axs[act_i].hist(durations, range=plot_range, bins=20, density=False, color=plot_color, alpha=0.6, label=plot_label)
      axs[act_i].set_title(activity)

  fig.suptitle('Event Duration Distributions', fontsize=16)
  fig.subplots_adjust(hspace=0.5)
  legend_elements = [Line2D([0], [0], color='blue', lw=4, label='Original'),
                   Line2D([0], [0], color='red', lw=4, label='Generated')]
  fig.legend(handles=legend_elements, loc='upper right')

  plt.savefig(os.path.join(output_path, output_filename))

  # reset plot settings to default and close figures
  plt.clf()
  plt.figure(figsize=plt.rcParams.get('figure.figsize'))
  plt.close(fig)
