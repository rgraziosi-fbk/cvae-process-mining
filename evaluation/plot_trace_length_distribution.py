import os
import torch
import matplotlib
from matplotlib import pyplot as plt

def plot_trace_length_distribution(original_dataset_path, generated_log_path, dataset_info=None, condition=None, output_path='output', output_filename='trace-length-distribution.png'):
  """
  Plot a histogram of trace lengths for original and generated datasets
  """

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  # Load logs
  original_dataset = dataset_info['CLASS'](
    dataset_path=original_dataset_path,
    max_trace_length=dataset_info['MAX_TRACE_LENGTH'],
    num_activities=dataset_info['NUM_ACTIVITIES'],
    num_labels=2,
    trace_attributes=dataset_info['TRACE_ATTRIBUTES'],
    activities=dataset_info['ACTIVITIES'],
    resources=dataset_info['RESOURCES'],
    activity_key=dataset_info['ACTIVITY_KEY'],
    resource_key=dataset_info['RESOURCE_KEY'],
    timestamp_key=dataset_info['TIMESTAMP_KEY'],
    trace_key=dataset_info['TRACE_KEY'],
    label_key=dataset_info['LABEL_KEY'],
  )
  
  generated_dataset = dataset_info['CLASS'](
    dataset_path=generated_log_path,
    max_trace_length=dataset_info['MAX_TRACE_LENGTH'],
    num_activities=dataset_info['NUM_ACTIVITIES'],
    num_labels=2,
    trace_attributes=dataset_info['TRACE_ATTRIBUTES'],
    activities=dataset_info['ACTIVITIES'],
    resources=dataset_info['RESOURCES'],
    activity_key=dataset_info['ACTIVITY_KEY'],
    timestamp_key=dataset_info['TIMESTAMP_KEY'],
    trace_key=dataset_info['TRACE_KEY'],
    label_key=dataset_info['LABEL_KEY'],
    highest_ts=original_dataset.highest_ts,
  )

  original_dataset_lens = []
  generated_dataset_lens = []

  for x, y in original_dataset:
    # if this trace doesn't have the specified label, skip it
    if condition is not None and not torch.equal(original_dataset.label2onehot[condition], y): continue
    
    _, acts, _, _ = x

    acts = acts.tolist()
    original_dataset_lens.append(acts.index(original_dataset.activity2n[original_dataset.EOT_ACTIVITY]))

  for x, y in generated_dataset:
    # if this trace doesn't have the specified label, skip it
    if condition is not None and not torch.equal(original_dataset.label2onehot[condition], y): continue

    _, acts, _, _ = x

    acts = acts.tolist()
    generated_dataset_lens.append(acts.index(generated_dataset.activity2n[generated_dataset.EOT_ACTIVITY]))

  matplotlib.use('agg')
  plt.clf()

  bins = [i for i in range(dataset_info['MAX_TRACE_LENGTH'])]
  plt.hist(original_dataset_lens, bins=bins, label=f'Original', color='blue', alpha=0.5, density=False)
  plt.hist(generated_dataset_lens, bins=bins, label='Generated', color='red', alpha=0.5, density=False)

  plt.title('Trace length distribution')
  plt.xlabel('Trace length')
  plt.ylabel('Num of traces')
  plt.legend(loc='upper right')

  plt.savefig(os.path.join(output_path, output_filename))