import os
import sys
import pandas as pd

from log_distance_measures.config import EventLogIDs, AbsoluteTimestampType, discretize_to_hour
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.work_in_progress import work_in_progress_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance

# our custom cwd metric
from evaluation.cwd_custom import circadian_workforce_distribution_distance_weekly

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

def compute_log_distance_measure(
  original_log_path,
  generated_log_path,
  dataset_info,
  measure='cfld',
  method=None,
  filter_log_by=None,
  cwd_resource_to_role_mapping_file=None,
  cwd_convert_resources_to_roles=True,
  gen_log_trace_key='case:concept:name',
  gen_log_activity_key='concept:name',
  gen_log_timestamp_key='time:timestamp',
  gen_resource_key='org:resource',
):
  original_log = read_log(original_log_path, separator=dataset_info['CSV_SEPARATOR'], verbose=False)
  generated_log = read_log(generated_log_path, separator=dataset_info['CSV_SEPARATOR'], verbose=False)

  # if provided, filter log by provided label
  if filter_log_by:
    original_log = original_log[original_log['label'] == filter_log_by]
    generated_log = generated_log[generated_log['label'] == filter_log_by]

  # if provided, load the resource-->role mapping csv file
  if cwd_resource_to_role_mapping_file:
    resource_role_mapping = pd.read_csv(cwd_resource_to_role_mapping_file, sep=',')

  original_log_ids = EventLogIDs(
    case=dataset_info['TRACE_KEY'],
    activity=dataset_info['ACTIVITY_KEY'],
    start_time='time:timestamp',
    end_time='time:timestamp',
    resource=dataset_info['RESOURCE_KEY'],
  )
  generated_log_ids = EventLogIDs(
    case=gen_log_trace_key,
    activity=gen_log_activity_key,
    start_time=gen_log_timestamp_key,
    end_time=gen_log_timestamp_key,
    resource=gen_resource_key,
  )

  generated_log[gen_log_timestamp_key] = generated_log[gen_log_timestamp_key].dt.tz_localize(None)
  
  if measure == 'cfld':
    original_log_traces = original_log[dataset_info['TRACE_KEY']].unique().tolist()
    generated_log_traces = generated_log[gen_log_trace_key].unique().tolist()
    # assert len(original_log_traces) == len(generated_log_traces)

    # if logs are not of the same size, drop cases to get same size
    if len(original_log_traces) > len(generated_log_traces):
      print(f'WARNING: Dropping cases from original log ({len(original_log_traces)}) to match the size of the generated log ({len(generated_log_traces)})')
      num_cases_to_drop = len(original_log_traces) - len(generated_log_traces)
      for _ in range(num_cases_to_drop):
        original_log_traces.pop()
      original_log = original_log[original_log[dataset_info['TRACE_KEY']].isin(original_log_traces)]
    elif len(generated_log_traces) > len(original_log_traces):
      print(f'WARNING: Dropping cases from generated log ({len(generated_log_traces)}) to match the size of the original log ({len(original_log_traces)})')
      num_cases_to_drop = len(generated_log_traces) - len(original_log_traces)
      for _ in range(num_cases_to_drop):
        generated_log_traces.pop()
      generated_log = generated_log[generated_log[gen_log_trace_key].isin(generated_log_traces)]

    return control_flow_log_distance(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
      parallel=False,
    )
  
  if 'ngram' in measure:
    ngram_n = int(measure.split('_')[-1])

    return n_gram_distribution_distance(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
      n=ngram_n,
      normalize=True,
    )
  
  if measure == 'aed':
    return absolute_event_distribution_distance(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
      discretize_type=AbsoluteTimestampType.START,
      discretize_event=discretize_to_hour,
    )
  
  if measure == 'cad':
    return case_arrival_distribution_distance(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
      discretize_event=discretize_to_hour,
    )
  
  if measure == 'ced':
    return circadian_event_distribution_distance(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
      discretize_type=AbsoluteTimestampType.START,
    )
  
  if measure == 'red':
    return relative_event_distribution_distance(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
      discretize_type=AbsoluteTimestampType.START,
      discretize_event=discretize_to_hour,
    )
  
  if measure == 'wip':
    return work_in_progress_distance(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
      window_size=pd.Timedelta(hours=1),
    )
  
  if measure == 'ctd':
    return cycle_time_distribution_distance(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
      bin_size=pd.Timedelta(hours=1),
    )
  
  if measure == 'cwd':
    # ensure resource column are of the same type
    original_log[dataset_info['RESOURCE_KEY']] = original_log[dataset_info['RESOURCE_KEY']].astype(str)
    if method in ['LSTM_1', 'LSTM_2']:
      generated_log['role'] = generated_log['role'].astype(str)
    else:
      generated_log[gen_resource_key] = generated_log[gen_resource_key].astype(str)

    # convert original_log resources to roles, if needed
    if cwd_convert_resources_to_roles:
      original_log = original_log.merge(
        resource_role_mapping,
        how='left',
        left_on=dataset_info['RESOURCE_KEY'],
        right_on='user'
      )
      original_log = original_log.drop(columns=[dataset_info['RESOURCE_KEY'], 'user'])
      original_log = original_log.rename(columns={'role': dataset_info['RESOURCE_KEY']})

      if method in ['LSTM_1', 'LSTM_2']:
        # if log generated by LSTM, just rename column
        generated_log = generated_log.rename(columns={'role': gen_resource_key})
      else:
        # otherwise map resources to roles
        generated_log = generated_log.merge(
          resource_role_mapping,
          how='left',
          left_on=gen_resource_key,
          right_on='user'
        )
        generated_log = generated_log.drop(columns=[gen_resource_key, 'user'])
        generated_log = generated_log.rename(columns={'role': gen_resource_key})

    print('[WARNING] Computing custom CWD metric. Change code if you want to use the default one.')
    return circadian_workforce_distribution_distance_weekly(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
    )