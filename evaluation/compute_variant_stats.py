import pm4py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log

def compute_variant_stats(train_log_path, test_log_path, generated_log_path, dataset_info=None, condition=None):  
  # Load logs
  train_log = read_log(train_log_path, separator=dataset_info['CSV_SEPARATOR'], verbose=False)
  test_log = read_log(test_log_path, separator=dataset_info['CSV_SEPARATOR'], verbose=False)
  generated_log = read_log(generated_log_path, separator=dataset_info['CSV_SEPARATOR'], verbose=False)
  
  if condition:
    train_log = train_log[train_log[dataset_info['LABEL_KEY']] == condition]
    test_log = test_log[test_log[dataset_info['LABEL_KEY']] == condition]
    generated_log = generated_log[generated_log[dataset_info['LABEL_KEY']] == condition]
  
  train_log[dataset_info['TRACE_KEY']] = train_log[dataset_info['TRACE_KEY']].astype(str)
  test_log[dataset_info['TRACE_KEY']] = test_log[dataset_info['TRACE_KEY']].astype(str)
  
  if dataset_info['TRACE_KEY'] in generated_log.columns:
    generated_log[dataset_info['TRACE_KEY']] = generated_log[dataset_info['TRACE_KEY']].astype(str)

  if 'case:concept:name' in generated_log.columns:
    generated_log['case:concept:name'] = generated_log['case:concept:name'].astype(str)

  train_variants = pm4py.stats.get_variants_as_tuples(
    train_log,
    timestamp_key='time:timestamp',
    activity_key=dataset_info['ACTIVITY_KEY'],
    case_id_key=dataset_info['TRACE_KEY'],
  )
  test_variants = pm4py.stats.get_variants_as_tuples(
    test_log,
    timestamp_key='time:timestamp',
    activity_key=dataset_info['ACTIVITY_KEY'],
    case_id_key=dataset_info['TRACE_KEY'],
  )
  gen_case_id_key = 'case:concept:name' if 'case:concept:name' in generated_log.columns else dataset_info['TRACE_KEY']
  gen_activity_id_key = 'concept:name' if 'concept:name' in generated_log.columns else dataset_info['ACTIVITY_KEY']
  generated_variants = pm4py.stats.get_variants_as_tuples(
    generated_log,
    timestamp_key='time:timestamp',
    activity_key=gen_activity_id_key,
    case_id_key=gen_case_id_key,
  )

  num_shared_variants_train = 0
  num_shared_traces_train = 0
  gen_variants_present_in_train = []

  # Count number of variants present in both train and generated log
  for train_variant, _ in train_variants.items():
    for generated_variant, num_generated in generated_variants.items():
      if train_variant == generated_variant:
        gen_variants_present_in_train.append(generated_variant)
        num_shared_variants_train += 1
        num_shared_traces_train += num_generated
        break

  # Count number of variants present in both test and generated log
  num_shared_variants_test = 0
  for test_variant, _ in test_variants.items():
    for generated_variant, num_generated in generated_variants.items():
      if test_variant == generated_variant:
        num_shared_variants_test += 1
        break

  return {
    'len_train_log': len(list(train_log[dataset_info['TRACE_KEY']].unique())),
    'len_generated_log': len(list(generated_log[gen_case_id_key].unique())),
    'num_train_variants': len(train_variants),
    'num_test_variants': len(test_variants),
    'num_generated_variants': len(generated_variants),
    'num_shared_variants_train': num_shared_variants_train,
    'num_shared_traces_train': num_shared_traces_train,
    'num_new_variants_train': len(generated_variants) - num_shared_variants_train,
    'num_shared_variants_test': num_shared_variants_test,
    'num_new_variants_test': len(generated_variants) - num_shared_variants_test
  }
