import os
import torch
import numpy as np
from torch.nn.functional import one_hot
from sklearn.manifold import TSNE
import matplotlib
from matplotlib import pyplot as plt

def plot_tsne(
  original_dataset_path,
  generated_log_path,
  consider_timestamps=True,
  perplexity=30,
  max_og_0=-1,
  max_og_1=-1,
  max_gen_0=-1,
  max_gen_1=-1,
  dataset_info=None,
  output_path=None,
  plot_filename='tsne-plot.png',
  plot_title='t-SNE',
  seed=42):
  """
  Plot data from original dataset (divided in 'false' and 'true')
  and generated dataset (supposing it has only 1 class).

  Params:
  original_dataset_path: path to original dataset
  generated_log_path: path to generated dataset
  consider_timestamps: whether to consider timestamps when building 1d trace for plotting
  max_og_0: max number of traces to plot from original dataset with label 0
  max_og_1: max number of traces to plot from original dataset with label 1
  max_gen_0: max number of traces to plot from generated dataset with label 0
  max_gen_1: max number of traces to plot from generated dataset with label 1
  dataset_info: dictionary containing usual information about dataset
  seed: for reproducibility
  """

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  tsne_X = []
  tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)

  # Load logs
  original_dataset = dataset_info['CLASS'](
    dataset_path=original_dataset_path,
    max_trace_length=dataset_info['MAX_TRACE_LENGTH'],
    num_activities=dataset_info['NUM_ACTIVITIES'],
    labels=dataset_info['LABELS'],
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
    labels=dataset_info['LABELS'],
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

  conditions = list(original_dataset.label2onehot.keys())

  def get_1d_trace(x):
    t = []
    attrs, acts, ts, ress = x

    # add attributes
    for i in range(len(original_dataset.trace_attributes)):
      if original_dataset.trace_attributes[i]['type'] == 'categorical':
        t.extend(one_hot(attrs[original_dataset.trace_attributes[i]['name']].to(torch.int64), num_classes=len(original_dataset.s2i[original_dataset.trace_attributes[i]['name']])).view(-1).tolist())
      elif original_dataset.trace_attributes[i]['type'] == 'numerical':
        t.append(attrs[original_dataset.trace_attributes[i]['name']].item())
      else:
        raise Exception(f'Unknown trace attribute type: {original_dataset.trace_attributes[i]["type"]}')

    # add activities
    acts = one_hot(acts.to(torch.int64), num_classes=dataset_info['NUM_ACTIVITIES']+1).view(-1).tolist()
    t.extend(acts)

    # add ts
    if consider_timestamps:
      ts = ts.tolist()
      t.extend(ts)

    # TODO: add resources to tsne plot

    return t

  # Get traces to plot
  # original dataset, label = 0
  n = 0
  for x, y in original_dataset:
    if n == max_og_0: break
    if y.argmax(dim=0) != 0: continue
      
    t = get_1d_trace(x)
    tsne_X.append(t)
    n += 1
  
  i1 = len(tsne_X)

  # original dataset, label = 1
  n = 0
  for x, y in original_dataset:
    if n == max_og_1: break
    if y.argmax(dim=0) != 1: continue

    t = get_1d_trace(x)
    tsne_X.append(t)
    n += 1

  i2 = len(tsne_X)

  # generated dataset, label = 0
  n = 0
  for x, y in generated_dataset:
    if n == max_gen_0: break
    if y.argmax(dim=0) != 0: continue

    t = get_1d_trace(x)
    tsne_X.append(t)
    n += 1

  i3 = len(tsne_X)

  # generated dataset, label = 1
  n = 0
  for x, y in generated_dataset:
    if n == max_gen_1: break
    if y.argmax(dim=0) != 1: continue

    t = get_1d_trace(x)
    tsne_X.append(t)
    n += 1

  # Fit t-sne
  tsne_emb = tsne.fit_transform(np.array(tsne_X))

  # Plot t-sne
  matplotlib.use('agg')
  plt.clf()

  plt.scatter(tsne_emb[0:i1,0], tsne_emb[0:i1,1], label=f'Original ({conditions[0]})', color='blue', marker='o', alpha=0.6, s=10)
  plt.scatter(tsne_emb[i1:i2,0], tsne_emb[i1:i2,1], label=f'Original ({conditions[1]})', color='red', marker='o', alpha=0.6, s=10)
  plt.scatter(tsne_emb[i2:i3,0], tsne_emb[i2:i3,1], label=f'Generated ({conditions[0]})', color='blue', marker='x', alpha=0.6, s=10)
  plt.scatter(tsne_emb[i3:,0], tsne_emb[i3:,1], label=f'Generated ({conditions[1]})', color='red', marker='x', alpha=0.6, s=10)

  plt.title(plot_title)
  plt.legend(loc='upper right')

  plt.savefig(os.path.join(output_path, plot_filename))