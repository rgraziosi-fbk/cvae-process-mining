import os
import torch
import tempfile
import math
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from ray.train import Checkpoint, get_checkpoint, report, get_context

from model import VAE
from train_utils import get_weights_cycle_linear
from utils import move_to_device

def train(config, dataset_info={}, checkpoint_every=10, output_dir='output', device='cpu', seed=42):
  # Dataset
  generator = torch.Generator().manual_seed(seed)
  train_dataset = dataset_info['CLASS'](
    dataset_path=dataset_info['TRAIN'],
    max_trace_length=dataset_info['MAX_TRACE_LENGTH'],
    num_activities=dataset_info['NUM_ACTIVITIES'],
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
  train_loader = DataLoader(dataset=train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, generator=generator)

  # Dataset
  val_dataset = dataset_info['CLASS'](
    dataset_path=dataset_info['VAL'],
    max_trace_length=dataset_info['MAX_TRACE_LENGTH'],
    num_activities=dataset_info['NUM_ACTIVITIES'],
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
  val_loader = DataLoader(dataset=val_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, generator=generator)

  # Model
  model = VAE(
    trace_attributes=dataset_info['TRACE_ATTRIBUTES'],
    num_activities=dataset_info['NUM_ACTIVITIES'],
    num_resources=dataset_info['NUM_RESOURCES'],
    max_trace_length=dataset_info['MAX_TRACE_LENGTH'],
    num_lstm_layers=config['NUM_LSTM_LAYERS'],
    attr_e_dim=config['ATTR_E_DIM'],
    act_e_dim=config['ACT_E_DIM'],
    res_e_dim=config['RES_E_DIM'],
    cf_dim=config['CF_DIM'],
    c_dim=config['C_DIM'],
    z_dim=config['Z_DIM'],
    dropout_p=config['DROPOUT_P'],
    device=device,
  ).to(device)

  # Loss functions
  def reconstruction_loss_fn(x_rec, x):
    cat_attrs_loss, num_attrs_loss, cf_loss, ts_loss, res_loss = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

    cat_attr_loss_fn = nn.BCELoss(reduction='sum')
    num_attr_loss_fn = nn.MSELoss(reduction='sum')
    cf_loss_fn = nn.BCELoss(reduction='sum')
    ts_loss_fn = nn.MSELoss(reduction='sum')
    res_loss_fn = nn.BCELoss(reduction='sum')

    attrs_rec, acts_rec, ts_rec, ress_rec = x_rec
    attrs, acts, ts, ress = x

    # convert categorical attrs to one-hot encoding
    for attr_name, attr_val in attrs.items():
      if attr_name in train_dataset.s2i: # it is a categorical attr
        attrs[attr_name] = F.one_hot(attr_val.to(torch.int64), num_classes=len(train_dataset.s2i[attr_name])).to(torch.float32).to(device)

    # turn all PAD activities into EOT activities
    acts = torch.where(acts == train_dataset.activity2n[train_dataset.PADDING_ACTIVITY], train_dataset.activity2n[train_dataset.EOT_ACTIVITY], acts).to(device)
    # convert acts to one-hot encoding
    acts = F.one_hot(acts.to(torch.int64), num_classes=dataset_info['NUM_ACTIVITIES']).to(torch.float32)

    # attrs
    for attr_name, attr_val in attrs.items():
      if attr_name in train_dataset.s2i:
        cat_attrs_loss += cat_attr_loss_fn(attrs_rec[attr_name], attr_val)
      else:
        attrs_rec[attr_name] = attrs_rec[attr_name].squeeze()
        num_attrs_loss += num_attr_loss_fn(attrs_rec[attr_name], attr_val).to(torch.float32)

    # acts
    cf_loss += cf_loss_fn(acts_rec, acts)

    # ts
    lens = acts.argmax(dim=2).argmax(dim=1)
    # use lens and compute loss only for the first lens elements
    for i in range(len(lens)):
      ts_rec[i, lens[i]:] = 0.0
      
    ts_loss += ts_loss_fn(ts_rec, ts)

    # ress
    ress = torch.where(ress == train_dataset.resource2n[train_dataset.PADDING_RESOURCE], train_dataset.resource2n[train_dataset.EOT_RESOURCE], ress).to(device)
    ress = F.one_hot(ress.to(torch.int64), num_classes=dataset_info['NUM_RESOURCES']).to(torch.float32)

    res_loss += res_loss_fn(ress_rec, ress)

    # sum up loss components
    loss = cat_attrs_loss + num_attrs_loss + cf_loss + ts_loss + res_loss

    return loss, torch.tensor([cat_attrs_loss, num_attrs_loss, cf_loss, ts_loss, res_loss])

  def kl_divergence_loss_fn(mean, var):
    var = torch.clamp(var, min=1e-9)
    return -torch.sum(1 + torch.log(var**2) - mean**2 - var**2)
  
  # Optimizer
  optimizer = optim.Adam(model.parameters(), lr=config['LR'])

  # Checkpoint loading
  checkpoint = get_checkpoint()

  if checkpoint:
    checkpoint_state = checkpoint.to_dict()

    start_epoch = checkpoint_state['epoch']
    current_best_val_loss = checkpoint_state['current_best_val_loss']
    model.load_state_dict(checkpoint_state['model_state_dict'])
    optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
  else:
    start_epoch = 0
    current_best_val_loss = float('inf')

  should_save_current_model = False
    
  w_kl = get_weights_cycle_linear(config['NUM_EPOCHS'], n_cycles=config['NUM_KL_ANNEALING_CYCLES'])

  # Setup directory for best models
  best_models_path = os.path.join(output_dir, 'best-models')
  os.makedirs(best_models_path, exist_ok=True)

  for epoch in range(start_epoch, config['NUM_EPOCHS']):
    # Training
    train_loss = 0.0
    train_rec_loss = 0.0
    train_kl_loss = 0.0
    train_rec_loss_components = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    ts_num_conformant = 0

    for x, y in train_loader:
      x, y = move_to_device(x, device), y.to(device)

      x_rec, mean, var = model(x, y)
      x_rec, mean, var = move_to_device(x_rec, device), move_to_device(mean, device), move_to_device(var, device)

      rec_loss, rec_loss_components = reconstruction_loss_fn(x_rec, x)
      kl_loss = kl_divergence_loss_fn(mean, var)
      loss = rec_loss + w_kl[epoch]*kl_loss

      train_loss += loss.item()
      train_rec_loss += rec_loss.item()
      train_kl_loss += kl_loss.item()
      train_rec_loss_components += rec_loss_components

      optimizer.zero_grad()
      loss.backward()

      optimizer.step()

      # compute number of traces with monotonically increasing ts
      _, _, ts_rec, _ = x_rec
      ts_num_conformant += torch.sum(ts_rec >= -0.000001).item()

    train_loss /= len(train_dataset)
    train_rec_loss /= len(train_dataset)
    train_kl_loss /= len(train_dataset)
    train_rec_loss_components /= len(train_dataset)
    ts_perc_conformant =  ts_num_conformant / (len(train_dataset) * dataset_info['MAX_TRACE_LENGTH'])

    # Validation
    val_loss = 0.0
    with torch.no_grad():
      for x, y in val_loader:
        x, y = move_to_device(x, device), y.to(device)

        x_rec, mean, var = model(x, y)
        x_rec, mean, var = move_to_device(x_rec, device), move_to_device(mean, device), move_to_device(var, device)

        rec_loss, rec_loss_components = reconstruction_loss_fn(x_rec, x)
        kl_loss = kl_divergence_loss_fn(mean, var)
        loss = rec_loss + w_kl[epoch]*kl_loss

        val_loss += loss.item()

    val_loss /= len(val_dataset)

    if math.isclose(w_kl[epoch], 1.0) and val_loss < current_best_val_loss:
      current_best_val_loss = val_loss
      should_save_current_model = True

    # Checkpoint save
    checkpoint_data = {
      'epoch': epoch,
      'current_best_val_loss': current_best_val_loss,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }

    if should_save_current_model:
      best_model_path = os.path.join(best_models_path, f'best-model-epoch-{epoch+1}.pt')
      torch.save(checkpoint_data, best_model_path)
      print(f'New best model saved at {best_model_path}')
      should_save_current_model = False
    
    session_report = {
      'val_loss': val_loss,
      'loss': train_loss,
      'rec_loss': train_rec_loss,
      'kl_loss': train_kl_loss,
      
      'w_kl': w_kl[epoch],

      # rec_loss components 
      'rec_cat_attrs_loss': train_rec_loss_components[0].item(),
      'rec_num_attrs_loss': train_rec_loss_components[1].item(),
      'rec_cf_loss': train_rec_loss_components[2].item(),
      'rec_ts_loss': train_rec_loss_components[3].item(),
      'rec_res_loss': train_rec_loss_components[4].item(),

      'rec_ts_perc_conformant': ts_perc_conformant,
    }

    # if it's a checkpoint
    # Save raytune checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
      if (epoch+1) % checkpoint_every == 0:
        experiment_path = get_context().get_trial_dir()

        # Save pytorch model
        checkpoint_path = os.path.join(experiment_path, 'checkpoints')
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(checkpoint_data, os.path.join(checkpoint_path, f'checkpoint-{epoch+1}.pt'))

        # Save raytune checkpoint
        checkpoint = Checkpoint.from_directory(tmpdir)
      else:
        checkpoint = None

      # ray tune report  
      report(metrics=session_report, checkpoint=checkpoint)
  
  print(f'Finished training with config = {config}')
