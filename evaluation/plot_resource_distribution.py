import os
import sys
from matplotlib import pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_log
from utils import get_dataset_attributes_info


def plot_resource_distribution(
  original_log_path,
  generated_log_path,
  dataset_info=None,
  cwd_resource_to_role_mapping_file=None,
  output_path='output',
  output_filename='resource-distribution.png',
):
  if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

  # If provided, load the resource -> role mapping csv file
  if cwd_resource_to_role_mapping_file:
    print('[WARN] Plotting resource distribution considering ROLES instead of RESOURCES!')

    resource_role_mapping = pd.read_csv(cwd_resource_to_role_mapping_file, sep=',')

  # Load logs
  original_log = read_log(
    original_log_path,
    separator=dataset_info['CSV_SEPARATOR'],
    verbose=False
  )

  generated_log = read_log(
    generated_log_path,
    separator=dataset_info['CSV_SEPARATOR'],
    verbose=False
  )

  original_log[dataset_info['RESOURCE_KEY']] = original_log[dataset_info['RESOURCE_KEY']].astype(str)
  generated_log[dataset_info['RESOURCE_KEY']] = generated_log[dataset_info['RESOURCE_KEY']].astype(str)

  # If needed, map resources to roles
  if cwd_resource_to_role_mapping_file:
    # ... in the original log ...
    original_log = original_log.merge(
      resource_role_mapping,
      how='left',
      left_on=dataset_info['RESOURCE_KEY'],
      right_on='user'
    )
    original_log = original_log.drop(columns=[dataset_info['RESOURCE_KEY'], 'user'])
    original_log = original_log.rename(columns={'role': dataset_info['RESOURCE_KEY']})

    # ... and in the generated log
    generated_log = generated_log.merge(
      resource_role_mapping,
      how='left',
      left_on=dataset_info['RESOURCE_KEY'],
      right_on='user'
    )
    generated_log = generated_log.drop(columns=[dataset_info['RESOURCE_KEY'], 'user'])
    generated_log = generated_log.rename(columns={'role': dataset_info['RESOURCE_KEY']})

  if cwd_resource_to_role_mapping_file is None:
    resources = dataset_info['RESOURCES']
    resources = sorted(resources)
  else:
    resources = list(set(resource_role_mapping['role'].tolist()))
    resources = sorted(resources, key=lambda x: int(x.split()[1])) # roles are written in this form 'Role 1', so we sort numerically by role number

  # From now on, 'resources' will be either actual resources, or roles if 'cwd_resource_to_role_mapping_file' was provided

  original_resource_distribution = original_log[dataset_info['RESOURCE_KEY']].value_counts(normalize=True)
  generated_resource_distribution = generated_log[dataset_info['RESOURCE_KEY']].value_counts(normalize=True)

  # add missing resources
  for resource in resources:
    if resource not in original_resource_distribution.keys().tolist():
      original_resource_distribution[resource] = 0

    if resource not in generated_resource_distribution.keys().tolist():
      generated_resource_distribution[resource] = 0

  original_resource_distribution = original_resource_distribution.sort_index()
  generated_resource_distribution = generated_resource_distribution.sort_index()


  original_resource_distribution_list = original_resource_distribution.tolist()
  generated_resource_distribution_list = generated_resource_distribution.tolist()

  assert len(resources) == len(original_resource_distribution_list) == len(generated_resource_distribution_list)

  # plot a single histogram with the two distributions
  fig, ax = plt.subplots()
  ax.bar(resources, original_resource_distribution_list, alpha=0.5, label='Original (train_log)', color='darkgreen')
  ax.bar(resources, generated_resource_distribution_list, alpha=0.5, label='Generated', color='red')
  ax.set_xlabel('Resource')
  ax.set_ylabel('Frequency')
  ax.legend()
  # plt.title('Resource distribution')
  plt.xticks(rotation=0)
  plt.tight_layout()

  plt.savefig(os.path.join(output_path, output_filename))

  plt.clf()
  plt.figure(figsize=plt.rcParams.get('figure.figsize'))
  plt.close(fig)



if __name__ == '__main__':
  LOG_NAME = 'bpic2012_b'
  
  log_path = f'/Users/riccardo/Documents/pdi/topics/data-augmentation/cvae-process-mining/datasets/{LOG_NAME}/{LOG_NAME}.csv'
  train_log_path = f'/Users/riccardo/Documents/pdi/topics/data-augmentation/cvae-process-mining/datasets/{LOG_NAME}/{LOG_NAME}_TRAIN.csv'
  test_log_path = f'/Users/riccardo/Documents/pdi/topics/data-augmentation/cvae-process-mining/datasets/{LOG_NAME}/{LOG_NAME}_TEST.csv'
  gen_log_path = f'/Users/riccardo/Documents/pdi/topics/data-augmentation/RESULTS/ProcessScienceCollection/cvae/{LOG_NAME}/generation_output/gen/gen1.csv'

  DATASET_INFO = {
    'FULL': log_path,
    'ACTIVITY_KEY': 'Activity',
    'RESOURCE_KEY': 'Resource',
    'TRACE_KEY': 'Case ID',
  }

  dataset_attributes_info = get_dataset_attributes_info(
    DATASET_INFO['FULL'],
    activity_key=DATASET_INFO['ACTIVITY_KEY'],
    trace_key=DATASET_INFO['TRACE_KEY'],
    resource_key=DATASET_INFO['RESOURCE_KEY'],
  )

  plot_resource_distribution(
    # test_log_path,
    train_log_path,
    gen_log_path,
    dataset_info={
      'CSV_SEPARATOR': ';',
      'RESOURCE_KEY': 'Resource',
      'RESOURCES': dataset_attributes_info['resources'],
    }
  )