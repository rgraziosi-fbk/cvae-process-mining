import torch
import datetime
import sys
import os
import pm4py
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess_log import add_relative_timestamp_between_activities, add_trace_attr_relative_timestamp_to_first_activity

# Generate data from given mean and var
def generate(model, mean, var, c=None):
  z = mean + var*torch.randn_like(var)
  z = z.view(1, -1)
  attrs, acts, ts, ress = model.decode(z, c)
  
  return attrs, acts, ts, ress


def generate_log(
  model,
  generation_config={},
  dataset_info={},
  output_path='output',
  output_name='gen',
):
  model.eval()

  if output_path and not os.path.exists(output_path):
    os.makedirs(output_path)

  dataset = dataset_info['CLASS'](
    dataset_path=dataset_info['FULL'],
    max_trace_length=dataset_info['MAX_TRACE_LENGTH'],
    num_activities=dataset_info['NUM_ACTIVITIES'],
    labels=dataset_info['LABELS'],
    num_labels=dataset_info['NUM_LABELS'],
    trace_attributes=dataset_info['TRACE_ATTRIBUTES'],
    activities=dataset_info['ACTIVITIES'],
    resources=dataset_info['RESOURCES'],
    activity_key=dataset_info['ACTIVITY_KEY'],
    timestamp_key=dataset_info['TIMESTAMP_KEY'],
    resource_key=dataset_info['RESOURCE_KEY'],
    trace_key=dataset_info['TRACE_KEY'],
    label_key=dataset_info['LABEL_KEY'],
  )

  test_dataset = dataset_info['CLASS'](
    dataset_path=dataset_info['TEST'],
    max_trace_length=dataset_info['MAX_TRACE_LENGTH'],
    num_activities=dataset_info['NUM_ACTIVITIES'],
    labels=dataset_info['LABELS'],
    num_labels=dataset_info['NUM_LABELS'],
    trace_attributes=dataset_info['TRACE_ATTRIBUTES'],
    activities=dataset_info['ACTIVITIES'],
    resources=dataset_info['RESOURCES'],
    activity_key=dataset_info['ACTIVITY_KEY'],
    timestamp_key=dataset_info['TIMESTAMP_KEY'],
    resource_key=dataset_info['RESOURCE_KEY'],
    trace_key=dataset_info['TRACE_KEY'],
    label_key=dataset_info['LABEL_KEY'],
  )

  generated = []
  ts_not_conformant_count = 0

  for condition, num_to_generate in generation_config.items():
    print(f'Generating {num_to_generate} traces with label "{condition}"...')

    condition_one_hot = None

    if model.is_conditional:
      condition_one_hot = dataset.label2onehot[condition].view(1, -1)

    for _ in range(num_to_generate):
      mean, var = torch.zeros((model.z_dim)), torch.ones((model.z_dim))
      attrs, acts, ts, ress = generate(model, mean, var, condition_one_hot)

      # count number of ts < 0, and set them to 0
      ts_not_conformant_count += torch.sum(ts < 0).item()
      ts[ts < 0] = 0

      # attrs
      trace_attrs = {}
      for attr in dataset.trace_attributes:
        if attr['type'] == 'categorical':
          attrs[attr['name']] = torch.argmax(attrs[attr['name']], dim=1)
        elif attr['type'] == 'numerical':
          attrs[attr['name']] = attr['min_value'] + (attrs[attr['name']] * (attr['max_value'] - attr['min_value'])) # min-max denormalization
        else:
          raise Exception(f'Unknown trace attribute type: {attr["type"]}')
        
        trace_attrs[attr['name']] = attrs[attr['name']]
      
      # acts
      acts = torch.argmax(acts, dim=2)

      # ress
      ress = torch.argmax(ress[:,:,:-1], dim=2) # exclude eot resource

      generated.append({
        'trace_attributes': trace_attrs,
        'activities': acts[0],
        'timestamps': ts[0],
        'resources': ress[0],
        'label': condition,
      })

  # print percentage of timestamps not conformant (i.e. < 0)
  tot_num_to_generate = sum(generation_config.values())
  ts_not_conformant_perc = (ts_not_conformant_count / (tot_num_to_generate*dataset_info['MAX_TRACE_LENGTH'])) * 100
  for out in [open(os.path.join(output_path, f'{output_name}-stats.txt'), mode='+a'), sys.stdout]:
    print(f'Number of timestamps non-conformant: {ts_not_conformant_count} ({ts_not_conformant_perc:.2f}%) (set to 0)', file=out)

  # Save generated data
  if output_path:
    new_data = []

    for i, generated_case in enumerate(generated):
      trace_attrs, activities, timestamps, resources, label = generated_case['trace_attributes'], generated_case['activities'], generated_case['timestamps'], generated_case['resources'], generated_case['label']

      start_datetime = test_dataset.log['time:timestamp'].min() + datetime.timedelta(minutes=trace_attrs['relative_timestamp_from_start'].item())
      current_datetime = start_datetime

      for j, (activity, resource) in enumerate(zip(activities, resources)):
        activity_name = dataset.n2activity[activity.item()]
        resource_name = dataset.n2resource[resource.item()]

        if activity_name == 'EOT':
          break

        cat_attrs = { attr_name: dataset.i2s[attr_name][attr_val.item()] for attr_name, attr_val in trace_attrs.items() if attr_name in dataset.i2s }
        num_attrs = { attr_name: attr_val.item() for attr_name, attr_val in trace_attrs.items() if attr_name not in dataset.i2s }
        
        relative_ts_minutes = (timestamps[j] * dataset.highest_ts).item()
        current_datetime = current_datetime + datetime.timedelta(minutes=relative_ts_minutes)
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        row = {
          **cat_attrs,
          **num_attrs,
          dataset_info['ACTIVITY_KEY']: activity_name,
          dataset_info['RESOURCE_KEY']: resource_name,
          dataset_info['TRACE_KEY']: f'GEN{i}',
          dataset_info['LABEL_KEY']: label,
          # also add these default columns
          'time:timestamp': formatted_datetime,
          'concept:name': activity_name,
          'org:resource': resource_name,
          'case:concept:name': f'GEN{i}',
          'case:label': label,
        }

        new_data.append(row)

    new_columns = dataset.log.columns.tolist()
    new_columns.extend(['concept:name', 'case:concept:name', 'case:label', 'org:resource']) # add default columns
    new_log = pd.DataFrame(new_data, columns=new_columns)
    new_log['time:timestamp'] = pd.to_datetime(new_log['time:timestamp'])

    # ensure case:concept:name column is of type str
    new_log[dataset_info['TRACE_KEY']] = new_log[dataset_info['TRACE_KEY']].astype(str)
    new_log['case:concept:name'] = new_log['case:concept:name'].astype(str)

    # add custom timestamp
    new_log = add_trace_attr_relative_timestamp_to_first_activity(new_log)
    new_log = add_relative_timestamp_between_activities(new_log)

    # save log as both xes and csv
    generated_dataset_path = os.path.join(output_path, f'{output_name}.xes')
    pm4py.write_xes(new_log, generated_dataset_path, case_id_key=dataset_info['TRACE_KEY'])
    new_log.to_csv(generated_dataset_path.replace('.xes', '.csv'), sep=dataset_info['CSV_SEPARATOR'])

    print(f'Generated log saved at {generated_dataset_path}')
    return generated_dataset_path
