import pm4py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

def split_random(log_path, log_name, output_path, split_perc=[0.7, 0.1, 0.2], csv_sep=';', case_id_key='case:concept:name'):
  assert len(split_perc) == 3
  assert sum(split_perc) == 1.0

  log = read_log(log_path, separator=csv_sep)
  train_perc, val_perc, test_perc = split_perc

  train, tmp = pm4py.ml.split_train_test(log, train_percentage=train_perc, case_id_key=case_id_key)
  val, test = pm4py.ml.split_train_test(tmp, train_percentage=val_perc/(val_perc+test_perc), case_id_key=case_id_key)

  total_cases = list(log.groupby(case_id_key).groups.values())
  train_cases = list(train.groupby(case_id_key).groups.values())
  val_cases = list(val.groupby(case_id_key).groups.values())
  test_cases = list(test.groupby(case_id_key).groups.values())

  assert len(train_cases) + len(val_cases) + len(test_cases) == len(total_cases)

  if log_path.endswith('.xes'):
    pm4py.write_xes(train, os.path.join(output_path, f'{log_name}_TRAIN.xes'))
    pm4py.write_xes(val, os.path.join(output_path, f'{log_name}_VAL.xes'))
    pm4py.write_xes(test, os.path.join(output_path, f'{log_name}_TEST.xes'))
  else:
    train.to_csv(os.path.join(output_path, f'{log_name}_TRAIN.csv'), sep=csv_sep)
    val.to_csv(os.path.join(output_path, f'{log_name}_VAL.csv'), sep=csv_sep)
    test.to_csv(os.path.join(output_path, f'{log_name}_TEST.csv'), sep=csv_sep)


def split_temporal(log_path, log_name, output_path, split_perc=[0.7, 0.1, 0.12], csv_sep=';', case_id_key='case:concept:name', timestamp_key='time:timestamp'):
  assert len(split_perc) == 3 # train, val, test
  assert 0.9999999999 <= sum(split_perc) <= 1.0000000001

  log = read_log(log_path, separator=csv_sep)
  train_perc, val_perc, test_perc = split_perc

  # sort cases by timestamp of first activity
  cases = list(log[case_id_key].unique())
  cases_by_start_time = { case: log[log[case_id_key] == case][timestamp_key].min() for case in cases }
  cases_by_start_time = sorted(cases_by_start_time.items(), key=lambda item: item[1])

  # split train-val-test
  train_cases = cases_by_start_time[:int(len(cases)*train_perc)]
  val_cases = cases_by_start_time[int(len(cases)*train_perc):int(len(cases)*(train_perc+val_perc))]
  test_cases = cases_by_start_time[int(len(cases)*(train_perc+val_perc)):]

  assert len(train_cases) + len(val_cases) + len(test_cases) == len(cases)

  train_cases = [train_case[0] for train_case in train_cases]
  train = log[log[case_id_key].isin(train_cases)]

  val_cases = [val_case[0] for val_case in val_cases]
  val = log[log[case_id_key].isin(val_cases)]

  test_cases = [test_case[0] for test_case in test_cases]
  test = log[log[case_id_key].isin(test_cases)]

  print(f'Train cases: {len(train_cases)}, Val cases: {len(val_cases)}, Test cases: {len(test_cases)}')

  if log_path.endswith('.xes'):
    pm4py.write_xes(train, os.path.join(output_path, f'{log_name}_TRAIN.xes'))
    pm4py.write_xes(val, os.path.join(output_path, f'{log_name}_VAL.xes'))
    pm4py.write_xes(test, os.path.join(output_path, f'{log_name}_TEST.xes'))
  else:
    train.to_csv(os.path.join(output_path, f'{log_name}_TRAIN.csv'), sep=csv_sep)
    val.to_csv(os.path.join(output_path, f'{log_name}_VAL.csv'), sep=csv_sep)
    test.to_csv(os.path.join(output_path, f'{log_name}_TEST.csv'), sep=csv_sep)


def split_temporal_arbitrary_num_splits(log_path, log_name, output_path, split_perc, csv_sep=';', case_id_key='case:concept:name', timestamp_key='time:timestamp'):
    assert 0.9999999999 <= sum(split_perc) <= 1.0000000001

    log = read_log(log_path, separator=csv_sep)

    # sort cases by timestamp of first activity
    cases = list(log[case_id_key].unique())
    cases_by_start_time = { case: log[log[case_id_key] == case][timestamp_key].min() for case in cases }
    cases_by_start_time = sorted(cases_by_start_time.items(), key=lambda item: item[1])

    # split cases
    splits = []
    start = 0
    for perc in split_perc:
      end = start + int(len(cases) * perc)
      split_cases = cases_by_start_time[start:end]
      split_cases = [case[0] for case in split_cases]
      split = log[log[case_id_key].isin(split_cases)]
      splits.append(split)
      start = end

    for i, split in enumerate(splits):
      split_cases = list(split[case_id_key].unique())
      print(f'Split {i+1} cases: {len(split_cases)}')

    # write splits to files
    for i, split in enumerate(splits):
        if log_path.endswith('.xes'):
            pm4py.write_xes(split, os.path.join(output_path, f'{log_name}_SPLIT_{i+1}.xes'))
        else:
            split.to_csv(os.path.join(output_path, f'{log_name}_SPLIT_{i+1}.csv'), sep=csv_sep)

    print(f'Split case counts: {[len(split) for split in splits]}')


def split_temporal_fixed_split_size(log_path, log_name, output_path, num_splits, split_size, csv_sep=';', case_id_key='case:concept:name', timestamp_key='time:timestamp'):
  log = read_log(log_path, separator=csv_sep)

  # sort cases by timestamp of first activity
  cases = list(log[case_id_key].unique())
  cases_by_start_time = { case: log[log[case_id_key] == case][timestamp_key].min() for case in cases }
  cases_by_start_time = sorted(cases_by_start_time.items(), key=lambda item: item[1])

  # split log into num_splits parts, each one containing split_size traces
  # splits may have overlapping traces, the important thing is that every split has same size
  splits = []
  start = 0
  step = (len(cases) - split_size) // (num_splits)
  for i in range(num_splits):
    end = start + split_size
    split_cases = cases_by_start_time[start:end]
    split_cases = [case[0] for case in split_cases]
    split = log[log[case_id_key].isin(split_cases)]
    splits.append(split)
    start += step

  for i, split in enumerate(splits):
    split_cases = list(split[case_id_key].unique())
    print(f'Split {i+1} cases: {len(split_cases)}')

  # write splits to files
  for i, split in enumerate(splits):
    if log_path.endswith('.xes'):
      pm4py.write_xes(split, os.path.join(output_path, f'{log_name}_SPLIT_{i+1}.xes'))
    else:
      split.to_csv(os.path.join(output_path, f'{log_name}_SPLIT_{i+1}.csv'), sep=csv_sep)

  print(f'Split case counts: {[len(split) for split in splits]}')


if __name__ == '__main__':
  split_temporal(
    os.path.join('datasets', 'sepsis_cases_1_custom.csv'),
    log_name='sepsis_cases_1_custom',
    output_path=os.path.join('datasets', 'sepsis_cases_1_custom'),
    split_perc=[0.7, 0.1, 0.2],
    case_id_key='Case ID',
  )

  # split_temporal_arbitrary_num_splits(
  #   os.path.join('datasets', 'sepsis_cases_1_cts.csv'),
  #   log_name='sepsis_cases_1_cts',
  #   output_path=os.path.join('datasets', 'sepsis_cases_1'),
  #   split_perc=[0.2 for i in range(5)],
  #   case_id_key='Case ID',
  # )

  # split_temporal_fixed_split_size(
  #   os.path.join('datasets', 'sepsis_cases_1_cts.csv'),
  #   log_name='sepsis_cases_1_cts',
  #   output_path=os.path.join('datasets', 'sepsis_cases_1'),
  #   num_splits=20,
  #   split_size=156,
  #   case_id_key='Case ID',
  # )
