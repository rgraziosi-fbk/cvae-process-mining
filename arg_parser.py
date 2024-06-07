import argparse
import os

def parse_arguments():
  parser = argparse.ArgumentParser(description='Conditional seq2seq VAE for process mining.')

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--cpu', action='store_true')
  parser.add_argument('--gpu', action='store_true')
  parser.add_argument('-d', '--dataset', action='store', dest='dataset_path',
                      help='Absolute path to dataset folder. The dataset must be present in this folder.',
                      default=os.path.abspath(os.path.join('.', 'datasets')))
  parser.add_argument('-o', '--output', action='store', dest='output_path',
                      help='Absolute path to output folder. Generated logs and plots will be saved in this folder.',
                      default=os.path.abspath(os.path.join('.', 'output')))
  parser.add_argument('-r', '--raytune', action='store', dest='raytune_path',
                      help='Absolute path to RayTune folder. Training metrics and checkpoints will be saved in this folder.',
                      default=os.path.abspath(os.path.join('.', 'raytune')))

  args = parser.parse_args()

  return args
