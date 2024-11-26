import sys
import os
import glob
from datetime import timedelta
import numpy as np
import pm4py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

# constants
CASE_ID_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
TIMESTAMP_KEY = 'time:timestamp'
LABEL_KEY = 'label'
RECOMPUTED_LABEL_KEY = 'recomputed_label'


def count_labels_in_log(labels, log_path, log=None, label_key='recomputed_label'):
  if log is None:
    log = read_log(log_path)

  cases = list(log[CASE_ID_KEY].unique())

  label_count = { label: 0 for label in labels }

  for case in cases:
    filtered_log = log[log[CASE_ID_KEY] == case]

    case_label = filtered_log[label_key].iloc[0]
    label_count[case_label] += 1

  # check that counted labels equals to the number of cases
  label_count_sum = 0
  for c in label_count.values():
    label_count_sum += c
  assert label_count_sum == len(cases)

  return label_count


### Methods for labeling the various logs ###

def label_sepsis(group, activity):
  # extract timestamp features
  group = group.sort_values(TIMESTAMP_KEY, ascending=False, kind='mergesort')
  tmp = group[TIMESTAMP_KEY] - group[TIMESTAMP_KEY].shift(-1)
  tmp = tmp.fillna(timedelta(0))
  group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

  # label cases
  relevant_activity_idxs = np.where(group[ACTIVITY_KEY] == activity)[0]
  if len(relevant_activity_idxs) > 0:
    idx = relevant_activity_idxs[0]
    if group["timesincelastevent"].iloc[idx] <= 28 * 1440: # return in less than 28 days
      group[RECOMPUTED_LABEL_KEY] = 'deviant'
    else:
      group[RECOMPUTED_LABEL_KEY] = 'regular'
  else:
    group[RECOMPUTED_LABEL_KEY] = 'regular'
  
  return group


def label_bpic2012_a(group):
  ACTIVITIES_FILTER = [
    "O_DECLINED-COMPLETE",
  ]
  filtered = pm4py.filter_event_attribute_values(
    group,
    attribute_key=ACTIVITY_KEY,
    values=ACTIVITIES_FILTER,
    level='case',
    retain=True,
    case_id_key=CASE_ID_KEY,
  )

  if len(filtered) == len(group):
    group[RECOMPUTED_LABEL_KEY] = 'deviant'
  elif len(filtered) == 0:
    group[RECOMPUTED_LABEL_KEY] = 'regular'
  else:
    assert False

  return group


def label_bpic2012_b(group):
  filtered = pm4py.filter_eventually_follows_relation(
    group,
    relations=[('O_CREATED-COMPLETE', 'O_CREATED-COMPLETE')],
    retain=True,
    case_id_key=CASE_ID_KEY,
    activity_key=ACTIVITY_KEY,
    timestamp_key=TIMESTAMP_KEY,
  )

  if len(filtered) == len(group):
    group[RECOMPUTED_LABEL_KEY] = 'deviant'
  elif len(filtered) == 0:
    group[RECOMPUTED_LABEL_KEY] = 'regular'
  else:
    assert False

  return group


def label_bpic2012_c(group):
  ACTIVITIES_FILTER = [
    "O_DECLINED-COMPLETE",
  ]
  DAYS_14 = 14 * 24 * 60 * 60

  recomputed_label = ''

  filtered = pm4py.filter_event_attribute_values(
    group,
    attribute_key=ACTIVITY_KEY,
    values=ACTIVITIES_FILTER,
    level='case',
    retain=True,
    case_id_key=CASE_ID_KEY,
  )

  if len(filtered) == len(group):
    recomputed_label += 'deviant'
  elif len(filtered) == 0:
    recomputed_label += 'regular'
  else:
    assert False

  recomputed_label += '_'

  filtered = pm4py.filter_case_performance(
    group,
    0,
    DAYS_14,
    timestamp_key=TIMESTAMP_KEY,
    case_id_key=CASE_ID_KEY,
  )

  if len(filtered) == len(group):
    recomputed_label += 'short'
  elif len(filtered) == 0:
    recomputed_label += 'long'
  else:
    assert False

  group[RECOMPUTED_LABEL_KEY] = recomputed_label

  return group


def label_traffic_fines(group):
  ACTIVITIES_FILTER = [
    "Appeal to Judge",
    "Insert Date Appeal to Prefecture",
    "Notify Result Appeal to Offender",
    "Receive Result Appeal from Prefecture",
    "Send Appeal to Prefecture",
  ]
  filtered = pm4py.filter_event_attribute_values(
    group,
    attribute_key='concept:name',
    values=ACTIVITIES_FILTER,
    level='case',
    retain=True,
    case_id_key=CASE_ID_KEY,
  )

  if len(filtered) == len(group):
    group[RECOMPUTED_LABEL_KEY] = 'deviant'
  elif len(filtered) == 0:
    group[RECOMPUTED_LABEL_KEY] = 'regular'
  else:
    assert False

  return group

### ... ###


def recompute_labels_on_generated_data(
  labels,
  generated_logs_path,
  log_name
):
  return_dict = { 'label_count': {}, 'num_errors': [] }
  label_count = { label: [] for label in labels }
  num_errors = []
  preds = {}

  for generated_log_path in generated_logs_path:
    generated_log = read_log(generated_log_path)

     # add a new column to log
    generated_log[RECOMPUTED_LABEL_KEY] = ""

    # prepare log
    generated_log = generated_log.sort_values(TIMESTAMP_KEY, ascending=True, kind="mergesort").groupby(CASE_ID_KEY)

    # recompute label based on log
    if log_name == 'sepsis':
      generated_log = generated_log.apply(label_sepsis, activity='Return ER')
    elif log_name == 'bpic2012_a':
      generated_log = generated_log.apply(label_bpic2012_a)
    elif log_name == 'bpic2012_b':
      generated_log = generated_log.apply(label_bpic2012_b)
    elif log_name == 'bpic2012_c':
      generated_log = generated_log.apply(label_bpic2012_c)
    elif log_name == 'traffic_fines':
      generated_log = generated_log.apply(label_traffic_fines)
    else:
      assert False, f'Unknown log name: {log_name}'
  
    # count number of errors between label and recomputed_label
    num_errors.append(len(list(generated_log[generated_log[LABEL_KEY] != generated_log[RECOMPUTED_LABEL_KEY]][CASE_ID_KEY].unique())))

    # count the number of predictions-ground_truth
    for label_predicted in labels:
      for label_ground_truth in labels:
        if f'{label_predicted}_{label_ground_truth}' not in preds:
          preds[f'{label_predicted}_{label_ground_truth}'] = [0]
        else:
          preds[f'{label_predicted}_{label_ground_truth}'].append(0)

    cases = generated_log[CASE_ID_KEY].unique().tolist()
    for case in cases:
      predicted_label = generated_log[generated_log[CASE_ID_KEY] == case][LABEL_KEY].iloc[0]
      ground_truth_label = generated_log[generated_log[CASE_ID_KEY] == case][RECOMPUTED_LABEL_KEY].iloc[0]
      
      preds[f'{predicted_label}_{ground_truth_label}'][-1] += 1

    # append label count
    log_label_count = count_labels_in_log(labels, generated_log_path, log=generated_log, label_key=RECOMPUTED_LABEL_KEY)
    for k, v in log_label_count.items():
      label_count[k].append(v)


  # Print some statistics
  num_errors_avg = sum(num_errors) / len(num_errors)
  num_errors_std = (sum([(x - num_errors_avg) ** 2 for x in num_errors]) / len(num_errors)) ** 0.5

  for label in labels:
    avg = sum(label_count[label]) / len(label_count[label])
    std = (sum([(x - avg) ** 2 for x in label_count[label]]) / len(label_count[label])) ** 0.5

    print(f'Num "{label}": {(avg):.2f} ± {(std):.2f}')
    return_dict[label] = { 'avg': avg, 'std': std }

  return_dict['predictions'] = {f'{label_predicted}_{label_ground_truth}': {'avg': None, 'std': None} for label_predicted in labels for label_ground_truth in labels}

  for label_predicted in labels:
    for label_ground_truth in labels:
      return_dict['predictions'][f'{label_predicted}_{label_ground_truth}']['avg'] = sum(preds[f'{label_predicted}_{label_ground_truth}']) / len(preds[f'{label_predicted}_{label_ground_truth}'])
      return_dict['predictions'][f'{label_predicted}_{label_ground_truth}']['std'] = (sum((x - return_dict['predictions'][f'{label_predicted}_{label_ground_truth}']['avg'])**2 for x in preds[f'{label_predicted}_{label_ground_truth}'])) / len(preds[f'{label_predicted}_{label_ground_truth}'])**0.5

  return_dict['num_errors'] = { 'avg': num_errors_avg, 'std': num_errors_std }
  return_dict['label_count'] = label_count

  return return_dict