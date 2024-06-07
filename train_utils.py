import numpy as np

# Cyclical annealing: https://github.com/haofuml/cyclical_annealing
def get_weights_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycles=4, ratio=0.5):
  L = np.ones(n_iter) * stop
  period = n_iter/n_cycles
  step = (stop-start)/(period*ratio) # linear schedule

  for c in range(n_cycles):
    v, i = start, 0
    
    while v <= stop and (int(i+c*period) < n_iter):
      L[int(i+c*period)] = v
      v += step
      i += 1
  
  return L