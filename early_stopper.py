import math
from ray.tune import Stopper

class EarlyStopper(Stopper):
  def __init__(self, patience=1, min_delta=0, debug=False):
    self.should_stop = False
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = float('inf')

    self.debug = debug

  def __call__(self, trial_id, result):
    if self.debug:
      with open('early-stop-debug.txt', mode='a') as f:
        print(f'[{result["training_iteration"]}] EarlyStopper: w_kl={result["w_kl"]}, val_loss={result["val_loss"]}; counter={self.counter}, should_stop={self.should_stop}', file=f)

    # never stop when not evaluating full loss
    if not math.isclose(result['w_kl'], 1): return False

    validation_loss = result['val_loss']

    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
    elif validation_loss > (self.min_validation_loss + self.min_delta):
      self.counter += 1

      if self.counter >= self.patience:
        self.should_stop = True

    if self.should_stop:
      print(f'Early stopping: validation loss has not improved for {self.patience} epochs')

    return self.should_stop

  def stop_all(self):
    return self.should_stop