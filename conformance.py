import os
import pm4py
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer

from utils import read_log

def convert_csv_log_to_xes(log_path, csv_separator=',', output_path='output', case_key='Case ID', activity_key='Activity', timestamp_key='time:timestamp'):
  log = read_log(log_path, separator=csv_separator)
  converted_log_path = os.path.join(output_path, 'log.xes')

  # Rename columns to standard names (necessary for Declare4Py)
  log.rename(columns={case_key: 'case:concept:name', activity_key: 'concept:name', timestamp_key: 'time:timestamp'}, inplace=True)

  # Ensure that 'case:concept:name' column is of type string (required by pm4py.write_xes)
  log['case:concept:name'] = log['case:concept:name'].astype(str)

  pm4py.write_xes(log, converted_log_path, case_id_key='case:concept:name')

  return converted_log_path


def discover_declare_model(log_path, log_csv_separator=',', case_key='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp', output_path='output',
                           consider_vacuity=False, min_support=0.8, itemsets_support=0.9, max_declare_cardinality=2, filter_log_by=None):
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  if log_path.endswith('.csv'):
    converted_log_path = convert_csv_log_to_xes(
      log_path,
      csv_separator=log_csv_separator,
      output_path=output_path,
      case_key=case_key,
      activity_key=activity_key,
      timestamp_key=timestamp_key,
    )

  if filter_log_by:
    converted_log_path = converted_log_path if converted_log_path is not None else log_path
    log_csv = read_log(converted_log_path)
    # log_csv = log.to_dataframe()
    log_csv = log_csv[log_csv['label'] == filter_log_by]
    converted_log_path_csv = converted_log_path.replace('.xes', '') + "_tmp.csv"
    log_csv.to_csv(converted_log_path_csv, sep=log_csv_separator, index=False)
    converted_log_path = convert_csv_log_to_xes(
      converted_log_path_csv,
      csv_separator=log_csv_separator,
      case_key=case_key,
      activity_key=activity_key,
      timestamp_key=timestamp_key,
    )

  event_log = D4PyEventLog(case_name='case:concept:name')
  event_log.parse_xes_log(log_path=converted_log_path if converted_log_path is not None else log_path)

  discovery = DeclareMiner(
    log=event_log,
    consider_vacuity=consider_vacuity,
    min_support=min_support,
    itemsets_support=itemsets_support,
    max_declare_cardinality=max_declare_cardinality
  )
  discovered_model = discovery.run()

  output_file = os.path.join(output_path, 'model.decl')
  discovered_model.to_file(output_file)

  # Temporary FIX: add a "|" at the end of each line of the file that contains a ","
  # This is necessary because d4py does not add a third "|" for binary constraints
  with open(output_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
      if ',' in line:
        lines[i] = line.strip() + ' |\n'
  
  with open(output_file, 'w') as f:
    f.writelines(lines)

  if converted_log_path is not None:
    os.remove(converted_log_path)

  return output_file


def conformance_checking(log_path, declare_model_path, consider_vacuity=False, filter_log_by=None, log_csv_separator=',', case_key='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp'):
  converted_log_path = None

  if log_path.endswith('.csv'):
    converted_log_path = convert_csv_log_to_xes(log_path, csv_separator=log_csv_separator, case_key=case_key, activity_key=activity_key, timestamp_key=timestamp_key)

  if filter_log_by and log_path.endswith('.csv'):
    converted_log_path = converted_log_path if converted_log_path is not None else log_path
    log_csv = read_log(converted_log_path)
    # log_csv = log.to_dataframe()
    log_csv = log_csv[log_csv['label'] == filter_log_by]
    converted_log_path_csv = converted_log_path.replace('.xes', '') + "_tmp.csv"
    log_csv.to_csv(converted_log_path_csv, sep=log_csv_separator, index=False)
    converted_log_path = convert_csv_log_to_xes(
      converted_log_path_csv,
      csv_separator=log_csv_separator,
      case_key=case_key,
      activity_key=activity_key,
      timestamp_key=timestamp_key,
    )
  elif filter_log_by and log_path.endswith('.xes'):
    xes_log = pm4py.read_xes(log_path, return_legacy_log_object=True)
    filtered_xes_log = pm4py.filter_event_attribute_values(xes_log, attribute_key='label', values=[filter_log_by], retain=True, level='case')
    converted_log_path = 'tmp/tmp.xes'
    pm4py.write_xes(filtered_xes_log, file_path=converted_log_path)

  event_log = D4PyEventLog(case_name='case:concept:name')
  event_log.parse_xes_log(log_path if converted_log_path == None else converted_log_path)

  declare_model = DeclareModel().parse_from_file(declare_model_path)

  basic_checker = MPDeclareAnalyzer(log=event_log, declare_model=declare_model, consider_vacuity=consider_vacuity)
  conf_check_res = basic_checker.run()

  if converted_log_path is not None:
    os.remove(converted_log_path)
  
  return conf_check_res.get_metric(metric='state')