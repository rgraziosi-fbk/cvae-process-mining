import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

def add_trace_attr_relative_timestamp_to_first_activity(
    log, trace_key='case:concept:name', timestamp_key='time:timestamp',
    custom_timestamp_key='relative_timestamp_from_start'):
  
  traces = list(log.groupby(trace_key).groups.values())
  log[custom_timestamp_key] = 0.0
  
  # get ts of first activity in the log
  lowest_timestamp = log[timestamp_key].min()

  for t in traces:
    # get ts of first activity in trace
    lowest_timestamp_trace = log.iloc[t][timestamp_key].min()
    # compute diff between first activity in trace and first activity in log
    custom_timestamp = (lowest_timestamp_trace - lowest_timestamp).total_seconds() / 60.0

    log.loc[t, custom_timestamp_key] = custom_timestamp

  return log


def add_relative_timestamp_between_activities(
    log, trace_key='case:concept:name', timestamp_key='time:timestamp',
    custom_timestamp_key='relative_timestamp_from_previous_activity'):
  
  traces = list(log.groupby(trace_key).groups.values())
  log[custom_timestamp_key] = 0.0

  for t in traces:
    for n, a in enumerate(t):
      if n == 0:
        log.loc[a, custom_timestamp_key] = 0.0
        continue

      log.loc[a, custom_timestamp_key] = (log.iloc[t[n]][timestamp_key] - log.iloc[t[n-1]][timestamp_key]).total_seconds() / 60.0

  return log



if __name__ == '__main__':
  DATASET = 'sepsis'
  DATASET_PATH = f'./datasets/{DATASET}/'
  DATASET_NAME = f'{DATASET}.csv'
  DATASET_TRACE_KEY = 'Case ID'
  DATASET_TIMESTAMP_KEY = 'time:timestamp'
  TIMESTAMP_FROM_START_KEY = 'relative_timestamp_from_start'
  TIMESTAMP_FROM_PREV_KEY = 'relative_timestamp_from_previous_activity'

  start_time = time.time()

  log = read_log(os.path.join(DATASET_PATH, DATASET_NAME), verbose=False)

  log = add_trace_attr_relative_timestamp_to_first_activity(
    log,
    trace_key=DATASET_TRACE_KEY,
    timestamp_key=DATASET_TIMESTAMP_KEY,
    custom_timestamp_key=TIMESTAMP_FROM_START_KEY,
  )
  log = add_relative_timestamp_between_activities(
    log,
    trace_key=DATASET_TRACE_KEY,
    timestamp_key=DATASET_TIMESTAMP_KEY,
    custom_timestamp_key=TIMESTAMP_FROM_PREV_KEY,
  )

  log.to_csv(os.path.join(DATASET_PATH,  f'{DATASET}_pp.csv'), sep=';', index=False)

  end_time = time.time()

  print(f'Execution time: {end_time - start_time} seconds')

  import matplotlib.pyplot as plt
  # Plot histogram of new timestamp
  plt.hist(log[TIMESTAMP_FROM_START_KEY], bins=50)
  plt.xlabel(TIMESTAMP_FROM_START_KEY)
  plt.ylabel('Frequency')
  plt.title(f'Histogram of {TIMESTAMP_FROM_START_KEY}')
  plt.show()

  plt.hist(log[TIMESTAMP_FROM_PREV_KEY], bins=50)
  plt.xlabel(TIMESTAMP_FROM_PREV_KEY)
  plt.ylabel('Frequency')
  plt.title(f'Histogram of {TIMESTAMP_FROM_PREV_KEY}')
  plt.show()
