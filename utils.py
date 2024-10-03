import os
import json
import pm4py
import torch
import pandas as pd
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from os import devnull

def read_log(dataset_path, separator=';', timestamp_key='time:timestamp', verbose=True):
  """Read xes or csv logs"""
  with suppress_stdout_stderr() if verbose is False else nullcontext():
    if dataset_path.endswith('.xes'):
      log = pm4py.read_xes(dataset_path)
    elif dataset_path.endswith('.csv'):
      log = pd.read_csv(dataset_path, sep=separator)
      log[timestamp_key] = pd.to_datetime(log[timestamp_key], format='mixed')
    else:
      raise ValueError("Unsupported file extension")
    
  return log


def get_dataset_attributes_info(
  dataset_path,
  activity_key='concept:name',
  trace_key='case:concept:name',
  resource_key='org:resource',
  trace_attributes=[],
):
  dataset_attributes_info = {}

  log = read_log(dataset_path, verbose=False)

  # Compute list of activities
  dataset_attributes_info['activities'] = log[activity_key].unique().tolist()

  # Compute list of resources
  resources = log[resource_key].unique().tolist()
  resources = [str(r) for r in resources]
  resources = list(set(resources))
  dataset_attributes_info['resources'] = resources

  # Compute max trace length
  traces = list(log.groupby(trace_key).groups.values())
  traces_lengths = [len(trace) for trace in traces]
  dataset_attributes_info['max_trace_length'] = max(traces_lengths)

  # Get info about each trace attribute
  dataset_attributes_info['trace_attributes'] = []
  for trace_attr in trace_attributes:
    possible_values = log[trace_attr].unique().tolist()
    possible_values.sort()
    is_numerical = all([isinstance(v, (int, float, complex)) for v in possible_values])

    trace_attribute_info = {
      'name': trace_attr,
      'type': 'numerical' if is_numerical else 'categorical',
    }

    if is_numerical:
      trace_attribute_info['min_value'] = min(possible_values)
      trace_attribute_info['max_value'] = max(possible_values)
    else:
      trace_attribute_info['possible_values'] = possible_values

    dataset_attributes_info['trace_attributes'].append(trace_attribute_info)

  return dataset_attributes_info


@contextmanager
def suppress_stdout_stderr():
  """A context manager that redirects stdout and stderr to devnull"""
  with open(devnull, 'w') as fnull:
    with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
      yield (err, out)


def move_to_device(data, device):
  """Move to specified device a list or dictionary of tensors"""
  if isinstance(data, list):
    return [move_to_device(item, device) for item in data]
  elif isinstance(data, dict):
    return {key: move_to_device(value, device) for key, value in data.items()}
  elif isinstance(data, torch.Tensor):
    return data.to(device)
  else:
    return data


def save_dict_to_json(d, filepath, indent=2):
  with open(filepath, mode='w') as f:
    json.dump(d, f, indent=indent)

def load_dict_from_json(filepath):
  with open(filepath, mode='r') as f:
    return json.load(f)