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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

def compute_log_distance_measure(
  original_log_path,
  generated_log_path,
  dataset_info,
  measure='cfld',
  filter_log_by=None,
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
    assert len(original_log_traces) == len(generated_log_traces)

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
    return circadian_workforce_distribution_distance(
      original_log,
      original_log_ids,
      generated_log,
      generated_log_ids,
    )