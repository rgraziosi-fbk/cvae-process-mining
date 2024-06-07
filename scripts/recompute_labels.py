import sys
import os
import glob
import numpy as np
import pm4py
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

LOG_NAME = 'traffic_fines_1_custom'

TRAIN_LOG_PATH = f'/Users/riccardo/Documents/pdi/topics/data-augmentation/dev/vae-gen-traces/datasets/{LOG_NAME}/traffic_fines_1_custom_TRAIN.csv'
TEST_LOG_PATH = f'/Users/riccardo/Documents/pdi/topics/data-augmentation/dev/vae-gen-traces/datasets/{LOG_NAME}/traffic_fines_1_custom_TEST_cropped.csv'
VAE_LOGS_PATH = glob.glob(f'/Users/riccardo/Desktop/RESULTS/FINAL/{LOG_NAME}/gen/' + '/*.csv')
LSTM_1_LOGS_PATH = glob.glob(f'/Users/riccardo/Desktop/RESULTS/FINAL/{LOG_NAME}/lstm_1/' + '/*.csv')
LSTM_2_LOGS_PATH = glob.glob(f'/Users/riccardo/Desktop/RESULTS/FINAL/{LOG_NAME}/lstm_2/' + '/*.csv')


pos_label = "deviant"
neg_label = "regular"
new_label_col = "label_new"
case_id_col = "case:concept:name"
activity_col = "concept:name"
timestamp_col = "time:timestamp"

def count_labels_in_log(log_path, log=None, case_id_key='case:concept:name', label_key='label'):
  if log is None:
    log = read_log(log_path)

  cases = list(log[case_id_key].unique())

  num_deviant, num_regular = 0, 0

  for case in cases:
    filtered_log = log[log[case_id_key] == case]

    if filtered_log[label_key].iloc[0] == pos_label:
      num_deviant += 1
    elif filtered_log[label_key].iloc[0] == neg_label:
      num_regular += 1
    else:
      assert False

  return num_deviant, num_regular


def label_sepsis_cases_1(group, activity):
  # extract timestamp features
  group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')
  tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
  tmp = tmp.fillna(timedelta(0))
  group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

  # label cases
  relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
  if len(relevant_activity_idxs) > 0:
    idx = relevant_activity_idxs[0]
    if group["timesincelastevent"].iloc[idx] <= 28 * 1440: # return in less than 28 days
      group[new_label_col] = pos_label
    else:
      group[new_label_col] = neg_label
  else:
    group[new_label_col] = neg_label
  
  return group


def label_bpic2012_2(group):
  ACTIVITIES_FILTER = [
    "O_DECLINED-COMPLETE",
  ]
  filtered = pm4py.filter_event_attribute_values(
    group,
    attribute_key=activity_col,
    values=ACTIVITIES_FILTER,
    level='case',
    retain=True,
    case_id_key=case_id_col,
  )

  if len(filtered) == len(group):
    group[new_label_col] = pos_label # deviant
  elif len(filtered) == 0:
    group[new_label_col] = neg_label # regular
  else:
    assert False

  return group


def label_bpic2012_2_custom(group):
  filtered = pm4py.filter_eventually_follows_relation(
    group,
    relations=[('O_CREATED-COMPLETE', 'O_CREATED-COMPLETE')],
    retain=True,
    case_id_key=case_id_col,
    activity_key=activity_col,
    timestamp_key=timestamp_col,
  )

  if len(filtered) == len(group):
    group[new_label_col] = pos_label # deviant
  elif len(filtered) == 0:
    group[new_label_col] = neg_label # regular
  else:
    assert False

  return group


def label_traffic_fines_1_custom(group):
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
    case_id_key=case_id_col,
  )

  if len(filtered) == len(group):
    group[new_label_col] = pos_label # deviant
  elif len(filtered) == 0:
    group[new_label_col] = neg_label # regular
  else:
    assert False

  return group

# # training
# train_num_deviant, train_num_regular = count_labels_in_log(TRAIN_LOG_PATH, case_id_key='Case ID')
# train_perc_deviant = train_num_deviant / (train_num_deviant + train_num_regular)

# print(f'(Training) Num deviant: {train_num_deviant}, Num regular: {train_num_regular}, Percentage deviant: {train_perc_deviant * 100:.2f}%, Percentage regular: {(1 - train_perc_deviant) * 100:.2f}%')

# # test
# test_num_deviant, test_num_regular = count_labels_in_log(TEST_LOG_PATH, case_id_key='Case ID')
# test_perc_deviant = test_num_deviant / (test_num_deviant + test_num_regular)

# print(f'(Test) Num deviant: {test_num_deviant}, Num regular: {test_num_regular}, Percentage deviant: {test_perc_deviant * 100:.2f}%, Percentage regular: {(1 - test_perc_deviant) * 100:.2f}%')

# # vae
# vae_num_deviant, vae_num_regular = [], []
# vae_num_errors = [] # number of traces mislabelled by VAE

# for vae_log_path in VAE_LOGS_PATH:
#   vae_log = read_log(vae_log_path)
  
#   # add a new col "label_new" to vae_log
#   vae_log[new_label_col] = ""

#   # recompute label
#   vae_log = vae_log.sort_values("time:timestamp", ascending=True, kind="mergesort").groupby("case:concept:name").apply(label_traffic_fines_1_custom)

#   # count the number of times "label" is different from "label_new" in dataframe
#   vae_num_errors.append(len(list(vae_log[vae_log["label"] != vae_log["label_new"]][case_id_col].unique())))

#   num_deviant, num_regular = count_labels_in_log(vae_log_path, log=vae_log, case_id_key='case:concept:name', label_key=new_label_col)
#   vae_num_deviant.append(num_deviant)
#   vae_num_regular.append(num_regular)

#   print(f'{vae_log_path}: Num deviant: {num_deviant}, Num regular: {num_regular}, Num errors: {vae_num_errors[-1]}')

# vae_num_deviant_avg = sum(vae_num_deviant) / len(vae_num_deviant)
# vae_num_regular_avg = sum(vae_num_regular) / len(vae_num_regular)
# vae_num_deviant_std = (sum([(x - vae_num_deviant_avg) ** 2 for x in vae_num_deviant]) / len(vae_num_deviant)) ** 0.5
# vae_num_regular_std = (sum([(x - vae_num_regular_avg) ** 2 for x in vae_num_regular]) / len(vae_num_regular)) ** 0.5
# vae_num_errors_avg = sum(vae_num_errors) / len(vae_num_errors)
# vae_num_errors_std = (sum([(x - vae_num_errors_avg) ** 2 for x in vae_num_errors]) / len(vae_num_errors)) ** 0.5

# vae_perc_deviant = vae_num_deviant_avg / (vae_num_deviant_avg + vae_num_regular_avg)

# print(f'(VAE) Num deviant: {vae_num_deviant_avg:.2f} ± {vae_num_deviant_std:.2f}, Num regular: {vae_num_regular_avg:.2f} ± {vae_num_regular_std:.2f}, Percentage deviant: {vae_perc_deviant * 100:.2f}%, Percentage regular: {(1 - vae_perc_deviant) * 100:.2f}%')
# print(f'(VAE) Num errors: {vae_num_errors_avg:.2f} ± {vae_num_errors_std:.2f}')

# lstm_1
lstm_1_num_deviant, lstm_1_num_regular = [], []
for lstm_1_log_path in LSTM_1_LOGS_PATH:
  lstm_1_log = read_log(lstm_1_log_path)

  # convert case:concept:name to string
  lstm_1_log['case:concept:name'] = lstm_1_log['case:concept:name'].astype(str)

  # add a new col "label_new" to vae_log
  lstm_1_log[new_label_col] = ""

  # recompute label
  lstm_1_log = lstm_1_log.sort_values("time:timestamp", ascending=True, kind="mergesort").groupby("case:concept:name").apply(label_traffic_fines_1_custom)

  # cannot compute errors since lstm_1 does not return a 'label'

  num_deviant, num_regular = count_labels_in_log(lstm_1_log_path, log=lstm_1_log, case_id_key='case:concept:name', label_key=new_label_col)
  lstm_1_num_deviant.append(num_deviant)
  lstm_1_num_regular.append(num_regular)

  print(f'{lstm_1_log_path}: Num deviant: {num_deviant}, Num regular: {num_regular}')

lstm_1_num_deviant_avg = sum(lstm_1_num_deviant) / len(lstm_1_num_deviant)
lstm_1_num_regular_avg = sum(lstm_1_num_regular) / len(lstm_1_num_regular)
lstm_1_num_deviant_std = (sum([(x - lstm_1_num_deviant_avg) ** 2 for x in lstm_1_num_deviant]) / len(lstm_1_num_deviant)) ** 0.5
lstm_1_num_regular_std = (sum([(x - lstm_1_num_regular_avg) ** 2 for x in lstm_1_num_regular]) / len(lstm_1_num_regular)) ** 0.5

lstm_1_perc_deviant = lstm_1_num_deviant_avg / (lstm_1_num_deviant_avg + lstm_1_num_regular_avg)

print(f'(LSTM_1) Num deviant: {lstm_1_num_deviant_avg:.2f} ± {lstm_1_num_deviant_std:.2f}, Num regular: {lstm_1_num_regular_avg:.2f} ± {lstm_1_num_regular_std:.2f}, Percentage deviant: {lstm_1_perc_deviant * 100:.2f}%, Percentage regular: {(1 - lstm_1_perc_deviant) * 100:.2f}%')

# lstm_2
lstm_2_num_deviant, lstm_2_num_regular = [], []
lstm_2_num_errors = [] # number of traces mislabelled by LSTM_2

for lstm_2_log_path in LSTM_2_LOGS_PATH:
  lstm_2_log = read_log(lstm_2_log_path)

  # convert case:concept:name to string
  lstm_2_log['case:concept:name'] = lstm_2_log['case:concept:name'].astype(str)

  # add a new col "label_new" to vae_log
  lstm_2_log[new_label_col] = ""

  # recompute label
  lstm_2_log = lstm_2_log.sort_values("time:timestamp", ascending=True, kind="mergesort").groupby("case:concept:name").apply(label_traffic_fines_1_custom)

  # count the number of times "label" is different from "label_new" in dataframe
  lstm_2_num_errors.append(len(list(lstm_2_log[lstm_2_log["label"] != lstm_2_log["label_new"]][case_id_col].unique())))

  num_deviant, num_regular = count_labels_in_log(lstm_2_log_path, log=lstm_2_log, case_id_key='case:concept:name', label_key=new_label_col)
  lstm_2_num_deviant.append(num_deviant)
  lstm_2_num_regular.append(num_regular)

  print(f'{lstm_2_log_path}: Num deviant: {num_deviant}, Num regular: {num_regular}, Num errors: {lstm_2_num_errors[-1]}')

lstm_2_num_deviant_avg = sum(lstm_2_num_deviant) / len(lstm_2_num_deviant)
lstm_2_num_regular_avg = sum(lstm_2_num_regular) / len(lstm_2_num_regular)
lstm_2_num_deviant_std = (sum([(x - lstm_2_num_deviant_avg) ** 2 for x in lstm_2_num_deviant]) / len(lstm_2_num_deviant)) ** 0.5
lstm_2_num_regular_std = (sum([(x - lstm_2_num_regular_avg) ** 2 for x in lstm_2_num_regular]) / len(lstm_2_num_regular)) ** 0.5

lstm_2_num_errors_avg = sum(lstm_2_num_errors) / len(lstm_2_num_errors)
lstm_2_num_errors_std = (sum([(x - lstm_2_num_errors_avg) ** 2 for x in lstm_2_num_errors]) / len(lstm_2_num_errors)) ** 0.5

lstm_2_perc_deviant = lstm_2_num_deviant_avg / (lstm_2_num_deviant_avg + lstm_2_num_regular_avg)

print(f'(LSTM_2) Num deviant: {lstm_2_num_deviant_avg:.2f} ± {lstm_2_num_deviant_std:.2f}, Num regular: {lstm_2_num_regular_avg:.2f} ± {lstm_2_num_regular_std:.2f}, Percentage deviant: {lstm_2_perc_deviant * 100:.2f}%, Percentage regular: {(1 - lstm_2_perc_deviant) * 100:.2f}%')
print(f'(LSTM_2) Num errors: {lstm_2_num_errors_avg:.2f} ± {lstm_2_num_errors_std:.2f}')