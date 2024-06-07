import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conformance import conformance_checking

def compute_conformance(
  log_path,
  declare_model_path,
  output_path,
  output_name,
  dataset_info,
  filter_log_by=None,
  consider_vacuity=False,
):
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  conformance_checking_result = conformance_checking(
    log_path,
    declare_model_path,
    consider_vacuity=consider_vacuity,
    filter_log_by=filter_log_by,
    log_csv_separator=dataset_info['CSV_SEPARATOR'],
    case_key=dataset_info['TRACE_KEY'],
    activity_key=dataset_info['ACTIVITY_KEY']
  )
  conformance_checking_result.to_csv(os.path.join(output_path, f'conformance_checking_result_{output_name}.csv'))
  conformance_per_trace = conformance_checking_result.mean(axis=1)
  conformance_per_trace.to_csv(os.path.join(output_path, f'conformance_checking_per_trace_result_{output_name}.csv'))
  
  return conformance_per_trace.mean()