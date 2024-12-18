import os
import math
from ray.tune import Stopper

class EarlyStopper(Stopper):
  def __init__(self, patience=1, min_delta_perc=0.0, output_dir='./output', debug=False):
    self.should_stop = False
    self.patience = patience
    self.min_delta_perc = min_delta_perc
    self.counter = 0
    self.min_validation_loss = float('inf')

    self.output_dir = output_dir
    self.debug = debug

  def __call__(self, trial_id, result):
    if self.debug:
      debug_file_path = os.path.join(self.output_dir, 'early-stop-debug.txt')

      with open(debug_file_path, mode='a') as f:
        print(f'[{trial_id}] [{result["training_iteration"]}] EarlyStopper: w_kl={result["w_kl"]:.2f}, val_loss={result["val_loss"]:.2f}; counter={self.counter}, min_delta_perc={self.min_delta_perc}, min_delta={(self.min_validation_loss*self.min_delta_perc):.2f}, should_stop={self.should_stop}', file=f)

    # never stop when not evaluating full loss
    if not math.isclose(result['w_kl'], 1):
      self.counter = 0 # also reset counter
      return False

    validation_loss = result['val_loss']

    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
    elif validation_loss > (self.min_validation_loss + self.min_delta_perc*self.min_validation_loss):
      self.counter += 1

      if self.counter >= self.patience:
        self.should_stop = True

    if self.should_stop:
      print(f'Early stopping: validation loss has not improved for {self.patience} epochs')

    return self.should_stop

  def stop_all(self):
    return self.should_stop