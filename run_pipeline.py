import os
import torch
import sys
import math
import shutil
from functools import partial
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig

from arg_parser import parse_arguments
from utils import get_dataset_attributes_info
from config import DATASET_INFO, MAX_TRACE_LENGTH, MAX_NUM_EPOCHS, NUM_SAMPLES, config
from train import train
from early_stopper import EarlyStopper

args = parse_arguments()

if args.debug:
  import ray
  ray.init(local_mode=True) # make RayTune debugging possible

# Config
if args.cpu and args.gpu:
  print('Choose either cpu or gpu or none.')
  exit(1)

if args.cpu:
  DEVICE = torch.device('cpu')
  DEVICE_STR = 'cpu'
elif args.gpu:
  DEVICE = torch.device('cuda')
  DEVICE_STR = 'cuda'
else:
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Using device: {DEVICE}')

MODEL_TYPE = 'cvae' # 'vae' or 'cvae'

shutil.copy('./config.py', 'output/config.py')

# Dataset
dataset_attributes_info = get_dataset_attributes_info(
  DATASET_INFO['FULL'],
  activity_key=DATASET_INFO['ACTIVITY_KEY'],
  trace_key=DATASET_INFO['TRACE_KEY'],
  trace_attributes=DATASET_INFO['TRACE_ATTRIBUTE_KEYS'],
)
max_trace_length = MAX_TRACE_LENGTH if MAX_TRACE_LENGTH else dataset_attributes_info['max_trace_length']

DATASET_INFO['MAX_TRACE_LENGTH'] = max_trace_length + 1
DATASET_INFO['ACTIVITIES'] = dataset_attributes_info['activities']
DATASET_INFO['NUM_ACTIVITIES'] = len(dataset_attributes_info['activities']) + 1
DATASET_INFO['TRACE_ATTRIBUTES'] = dataset_attributes_info['trace_attributes']

# Configure hyperparam search space
config['C_DIM'] = DATASET_INFO['NUM_LABELS'] if MODEL_TYPE == 'cvae' else 0
config['NUM_EPOCHS'] = MAX_NUM_EPOCHS

scheduler = ASHAScheduler(
  max_t=MAX_NUM_EPOCHS,
  grace_period=50,
  reduction_factor=2
)

trainable = partial(train, dataset_info=DATASET_INFO, checkpoint_every=50, tmp_path=config['TMP_PATH'], device=DEVICE)
if DEVICE_STR == 'cpu':
  trainable_with_resources = tune.with_resources(trainable, { "cpu": 1 })
else:
  trainable_with_resources = tune.with_resources(trainable, { "gpu": 1 })

tuner = tune.Tuner(
  trainable_with_resources,
  param_space=config,
  tune_config=tune.TuneConfig(
    metric='loss',
    mode='min',
    scheduler=scheduler,
    num_samples=NUM_SAMPLES,
  ),
  run_config=RunConfig(
    storage_path=args.raytune_path,
    stop=EarlyStopper(
      patience=config['EARLY_STOPPING_PATIENCE'],
      min_delta=config['EARLY_STOPPING_MIN_DELTA'],
      debug=config['EARLY_STOPPING_DEBUG'],
    ),
  ),
)

# Uncomment if you want to resume past training
# tuner = tune.Tuner.restore(
#   os.path.abspath(os.path.join(args.raytune_path, 'train_2023-10-13_09-22-46')),
#   trainable=partial(train, dataset_info=DATASET_INFO, checkpoint_every=50, device=DEVICE),
#   param_space=config,
#   resume_unfinished=True,
#   resume_errored=False,
# )

results = tuner.fit()

# Get best checkpoint (i.e. minimum loss value when evaluating full loss function, i.e. w_kl=1)
best_trial = results.get_best_result('loss', 'min')
best_checkpoint = None

for checkpoint in best_trial.best_checkpoints:
  path, metrics = checkpoint
  current_best_loss = best_checkpoint[1]['loss'] if best_checkpoint else math.inf
  
  if metrics['w_kl'] == 1 and metrics['loss'] < current_best_loss:
    best_checkpoint = checkpoint

best_checkpoint_path = best_checkpoint[0].path
best_checkpoint_metrics = best_checkpoint[1]

# Print best configuration
for out in [open(os.path.join(args.output_path, 'stats.txt'), mode='w'), sys.stdout]:
  print(f'Best trial checkpoint: {best_checkpoint_path}', file=out)
  print(f'Best trial config: {best_checkpoint_metrics["config"]}', file=out)
  print(f'Best trial final validation loss: {best_checkpoint_metrics["loss"]}', file=out)
