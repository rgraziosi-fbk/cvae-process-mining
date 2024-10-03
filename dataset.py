import torch
import copy
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from utils import read_log

class GenericDataset(Dataset):
  """
    dataset_path: path to the xes log
    max_trace_length: length of the longest trace in the log + 1 (for EOT activity)
    num_activities: number of different activities in the log + 1 (for EOT activity)
    num_labels: number of different labels in the log
    trace_key: name of the column of the log containing traces id
    activity_key: name of the column of the log containing activity names
    timestamp_key: name of the column of the log containing the timestamp
    resource_key: name of the column of the log containing the resource
    label_key: name of the column of the log containing the label
    activities: (optional) list of activities to consider (used to impose a specific list of activities instead of using the ones found in the provided log)
    resources: (optional) list of resources to consider (used to impose a specific list of resources instead of using the ones found in the provided log)
  """
  def __init__(
    self, dataset_path='', max_trace_length=100, num_activities=10, num_labels=2,
    trace_key='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp', resource_key='org:resource',
    label_key='case:label', trace_attributes=[], activities=None, resources=None, highest_ts=None,
  ):
    self.log = read_log(dataset_path, verbose=False)

    self.max_trace_length = max_trace_length
    self.num_activities = num_activities
    self.num_labels = num_labels
    self.trace_attributes = trace_attributes

    # Special activities
    self.EOT_ACTIVITY = 'EOT'
    self.PADDING_ACTIVITY = 'PAD'

    # Special resources
    self.EOT_RESOURCE = 'EOT-RES'
    self.PADDING_RESOURCE = 'PAD-RES'

    # Get activity names
    if activities is None:
      activities = self.log[activity_key].unique().tolist()
    else:
      activities = copy.deepcopy(activities)
    activities.sort()
    activities.append(self.EOT_ACTIVITY)
    activities.append(self.PADDING_ACTIVITY)

    assert len(activities)-1 == self.num_activities # Check whether num_activities and found number activities coincide

    # Mapping from activity name to index
    self.activity2n = { a: n for n, a in enumerate(activities) }
    self.n2activity = { n: a for n, a in enumerate(activities) }


    # Get resources
    if resources is None:
      resources = self.log[resource_key].unique().tolist()
    else:
      resources = copy.deepcopy(resources)

    # cast every item of resources to str
    resources = [str(r) for r in resources]

    resources.sort()
    resources.append(self.EOT_RESOURCE)
    resources.append(self.PADDING_RESOURCE)

    # Mapping from resource name to index
    self.resource2n = { r: n for n, r in enumerate(resources) }
    self.n2resource = { n: r for n, r in enumerate(resources) }


    # Get highest ts value
    self.highest_ts = self.log[timestamp_key].quantile(q=0.95) if highest_ts is None else highest_ts

    # Get label names
    labels = self.log[label_key].unique().tolist()
    labels.sort()

    assert len(labels) == self.num_labels

    # Mapping from label name to one hot encoding
    self.label2onehot = { label: one_hot(torch.tensor(i), num_classes=self.num_labels) for i, label in enumerate(labels) }

    # Build trace attributes mapping
    self.s2i, self.i2s = {}, {}
    for trace_attr in trace_attributes:
      if trace_attr['type'] == 'categorical':
        self.s2i[trace_attr['name']] = { a: n for n, a in enumerate(trace_attr['possible_values']) }
        self.i2s[trace_attr['name']] = { n: a for n, a in enumerate(trace_attr['possible_values']) }

    # Build dataset
    traces = list(self.log.groupby(trace_key).groups.values())
    self.x, self.y = [], []

    # trace attributes
    for trace in traces:
      x_attr = {}
      trace_idx = trace[0]
      for trace_attr in trace_attributes:
        attr = self.log.iloc[trace_idx][trace_attr['name']] # get attribute value

        if trace_attr['type'] == 'categorical':
          attr = self.s2i[trace_attr['name']][attr] # convert attribute to index
          attr = torch.tensor(attr, dtype=torch.int64)
        elif trace_attr['type'] == 'numerical':
          attr = (attr - trace_attr['min_value']) / (trace_attr['max_value'] - trace_attr['min_value']) # min-max normalization
          attr = torch.tensor(attr, dtype=torch.float32)
        else:
          raise Exception(f'Unknown trace attribute type: {trace_attr["type"]}')
        
        x_attr[trace_attr['name']] = attr

      # activities
      x_trace = self.log.iloc[trace][activity_key].tolist()[:self.max_trace_length-1] # get trace activities
      x_trace += [self.EOT_ACTIVITY] # append End Of Trace token
      x_trace += [self.PADDING_ACTIVITY] * (self.max_trace_length - len(trace) - 1) # append padding if needed
      x_trace = torch.tensor([self.activity2n[a] for a in x_trace]).to(torch.int) # convert to tensor

      # timestamps
      x_ts = self.log.iloc[trace][timestamp_key].tolist()[:self.max_trace_length-1] # get trace timestamps
      x_ts += [0] # append timestamp for fictional "End Of Trace" activity
      x_ts += [0] * (self.max_trace_length - len(trace) - 1) # append padding if needed
      x_ts = [ts / self.highest_ts for ts in x_ts] # normalize
      x_ts = torch.tensor(x_ts).to(torch.float32) # convert to tensor

      # resources
      x_res = self.log.iloc[trace][resource_key].tolist()[:self.max_trace_length-1]
      x_res += [self.EOT_RESOURCE]
      x_res += [self.PADDING_RESOURCE] * (self.max_trace_length - len(trace) - 1)
      x_res = [str(x) for x in x_res]
      x_res = torch.tensor([self.resource2n[r] for r in x_res]).to(torch.int)
      
      # label
      y = self.log.iloc[trace][label_key].tolist()[0] # get label from log
      y = self.label2onehot[y].to(torch.float32) # convert to one-hot tensor

      self.x.append((x_attr, x_trace, x_ts, x_res))
      self.y.append(y)
      
  def __len__(self):
    return len(self.x)
  
  def __getitem__(self, i):
    return self.x[i], self.y[i]
