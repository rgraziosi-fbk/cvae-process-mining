import os
import torch
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap # requires package 'umap-learn'

# Config

# LOG can be either: sepsis, traffic_fines, bpic2012_a
LOG = 'bpic2012_a'

# WHAT_TO_COMPUTE = ['distance', 'pca', 'tsne', 'umap', 'mds']
WHAT_TO_COMPUTE = ['tsne']
SHOULD_SHOW_PLOT = True

REMOVE_PAD = True

PLOT_ANNOTATION_OFFSET = 0.05

SEED = 42

INPUT_ROOT_PATH = os.path.join('embeddings_analysis', 'input')
OUTPUT_ROOT_PATH = os.path.join('embeddings_analysis', 'output')
EMBEDDINGS_TENSOR_PATH = os.path.join(INPUT_ROOT_PATH, f'{LOG}_act_embeddings.pt')
ACTIVITY2N_PATH = os.path.join(INPUT_ROOT_PATH, f'{LOG}_activity2n.json')

# Load embeddings data
embs = torch.load(EMBEDDINGS_TENSOR_PATH)
embs = embs.detach()

with open(ACTIVITY2N_PATH, 'r') as f:
  a2n = json.load(f)
acts = list(a2n.keys())

# Remove PAD activity if requested
if REMOVE_PAD:
  embs = embs[:-1,:]
  acts = acts[:-1]

assert embs.shape[0] == len(acts), "Number of activities and embeddings do not match"

# 1. Compute pairwise distance between activity embedding vectors and plot heatmap
if 'distance' in WHAT_TO_COMPUTE:
  distances = torch.cdist(embs, embs, p=2)

  plt.figure(figsize=(8, 6))
  plt.imshow(distances.numpy(), cmap="viridis", interpolation="nearest")

  plt.colorbar(label="Distance")

  plt.xticks(ticks=range(len(acts)), labels=acts, rotation=90)
  plt.yticks(ticks=range(len(acts)), labels=acts)

  plt.title("Pairwise Distance Heatmap")
  plt.xlabel("Activity")
  plt.ylabel("Activity")

  if SHOULD_SHOW_PLOT:
    plt.show()
  else:
    plt.savefig(os.path.join(OUTPUT_ROOT_PATH, f'{LOG}_pairwise_distance_heatmap.png'))

  plt.clf()


# 2. PCA
if 'pca' in WHAT_TO_COMPUTE:
  print('\nComputing PCA...')

  pca = PCA(n_components=2, random_state=SEED)
  embs_2d = pca.fit_transform(embs.numpy())

  print(f'Explained variance ratios: PC1 = {pca.explained_variance_ratio_[0]:.2f}, PC2 = {pca.explained_variance_ratio_[1]:.2f}')
  print(f'Total explained variance from the two components = {pca.explained_variance_ratio_.sum():.2f}')

  plt.scatter(embs_2d[:, 0], embs_2d[:, 1])
  for i, act in enumerate(acts):
    plt.annotate(act, (embs_2d[i, 0] + PLOT_ANNOTATION_OFFSET, embs_2d[i, 1] + PLOT_ANNOTATION_OFFSET))
  plt.title("PCA")
  
  if SHOULD_SHOW_PLOT:
    plt.show()
  else:
    plt.savefig(os.path.join(OUTPUT_ROOT_PATH, f'{LOG}_pca.png'))

  plt.clf()


# 3. t-SNE
if 'tsne' in WHAT_TO_COMPUTE:
  tsne = TSNE(n_components=2, perplexity=15, random_state=SEED)
  embs_2d = tsne.fit_transform(embs.numpy())

  plt.scatter(embs_2d[:, 0], embs_2d[:, 1])
  for i, act in enumerate(acts):
    plt.annotate(act, (embs_2d[i, 0] + PLOT_ANNOTATION_OFFSET, embs_2d[i, 1] + PLOT_ANNOTATION_OFFSET))
  plt.title("t-SNE")
  
  if SHOULD_SHOW_PLOT:
    plt.show()
  else:
    plt.savefig(os.path.join(OUTPUT_ROOT_PATH, f'{LOG}_tsne.png'))

  plt.clf()


# 4. umap
if 'umap' in WHAT_TO_COMPUTE:
  reducer = umap.UMAP(n_components=2, random_state=SEED)
  embs_2d = reducer.fit_transform(embs.numpy())

  plt.scatter(embs_2d[:, 0], embs_2d[:, 1])
  for i, word in enumerate(acts):
    plt.annotate(acts, (embs_2d[i, 0] + PLOT_ANNOTATION_OFFSET, embs_2d[i, 1] + PLOT_ANNOTATION_OFFSET))
  plt.title("UMAP")
  
  if SHOULD_SHOW_PLOT:
    plt.show()
  else:
    plt.savefig(os.path.join(OUTPUT_ROOT_PATH, f'{LOG}_umap.png'))

  plt.clf()


# 5. mds
if 'mds' in WHAT_TO_COMPUTE:
  mds = MDS(n_components=2, random_state=SEED)
  embs_2d = mds.fit_transform(embs.numpy())

  plt.scatter(embs_2d[:, 0], embs_2d[:, 1])
  for i, act in enumerate(acts):
    plt.annotate(act, (embs_2d[i, 0] + PLOT_ANNOTATION_OFFSET, embs_2d[i, 1] + PLOT_ANNOTATION_OFFSET))
  plt.title("MDS")
  
  if SHOULD_SHOW_PLOT:
    plt.show()
  else:
    plt.savefig(os.path.join(OUTPUT_ROOT_PATH, f'{LOG}_mds.png'))

  plt.clf()

print('\n')