import os
import glob
import copy
import time
from matplotlib import pyplot as plt

from evaluation.generate_log import generate_log
from evaluation.recompute_labels_on_generated_data import recompute_labels_on_generated_data
from evaluation.compute_conformance import compute_conformance
from evaluation.compute_log_distance_measure import compute_log_distance_measure
from evaluation.plot_boxplots import plot_boxplots
from evaluation.plot_tsne import plot_tsne
from evaluation.plot_trace_length_distribution import plot_trace_length_distribution
from evaluation.compute_variant_stats import compute_variant_stats
from evaluation.plot_activity_duration_distributions import plot_activity_duration_distributions
from evaluation.plot_resource_distribution import plot_resource_distribution
from evaluation.plot_activity_distribution_by_resource import plot_activity_distribution_by_resource
from evaluation.plot_trace_attribute_distributions import plot_trace_attribute_distributions

from conformance import discover_declare_model
from utils import save_dict_to_json

plt.set_loglevel('warning')

LOG_3_KEY = 'LOG_3'
CVAE_KEY = 'CVAE'
LSTM_1_KEY = 'LSTM_1'
LSTM_2_KEY = 'LSTM_2'
TRANSFORMER_1_KEY = 'PT_1'
TRANSFORMER_2_KEY = 'PT_2'
PROCESSGAN_1_KEY = 'PG_1'

def evaluate_generation(
  model,
  log_name,
  should_generate=True,
  generation_config={},
  dataset_info={},
  input_path=None,
  output_path=None,

  should_use_cvae=True,
  should_use_log_3=True,
  should_use_lstm_1=True,
  should_use_lstm_2=True,
  should_use_transformer_1=True,
  should_use_transformer_2=True,
  should_use_processgan_1=True,

  should_recompute_labels_on_generated_data=True,

  should_skip_all_metrics_computation=False,
  should_plot_boxplots=True,

  # conformance
  should_compute_conformance=True,
  consider_vacuity=False,
  min_support=0.9,
  itemsets_support=0.9,
  max_declare_cardinality=2,

  # log distance measures
  log_distance_measures_to_compute=[],
  log_distance_measures_also_compute_filtered_by=[],
  cwd_resource_to_role_mapping_file=None,

  # t-sne
  should_plot_tsne=True,
  tsne_max_og_0=-1,
  tsne_max_og_1=-1,
  tsne_max_gen_0=-1,
  tsne_max_gen_1=-1,

  # trace length distribution
  should_plot_trace_length_distribution=True,

  # variant statistics
  should_compute_variant_stats=True,

  # activity duration distribution
  should_plot_activity_duration_distributions=True,
  activity_duration_distributions_filter_by_label=None,

  # resources
  should_plot_resource_distribution=True,
  should_plot_activity_by_resource_distribution=True,

  # trace attribute distributions
  should_plot_trace_attribute_distributions=True,
  trace_attributes={},

  # other
  seed=42,
):
  """
  Generate log and evaluate it
  """
  model.eval()

  DEFAULT_METHODS_DICT = {
    **({LOG_3_KEY: []} if should_use_log_3 else {}),
    **({CVAE_KEY: []} if should_use_cvae else {}),
    **({LSTM_1_KEY: []} if should_use_lstm_1 else {}),
    **({LSTM_2_KEY: []} if should_use_lstm_2 else {}),
    **({TRANSFORMER_1_KEY: []} if should_use_transformer_1 else {}),
    **({TRANSFORMER_2_KEY: []} if should_use_transformer_2 else {}),
    **({PROCESSGAN_1_KEY: []} if should_use_processgan_1 else {}),
  }

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  log_paths_per_method = copy.deepcopy(DEFAULT_METHODS_DICT)

  # VAE
  if should_generate:
    gen_start_time = time.time()

    for i in range(generation_config['NUM_GENERATIONS']):
      print(f'Generating log {i+1}/{generation_config["NUM_GENERATIONS"]}')
      generated_log_path = generate_log(
        model,
        generation_config=generation_config['LABELS'],
        dataset_info=dataset_info,
        output_path=os.path.join(output_path, 'gen'),
        output_name=f'gen{i+1}'
      )
      log_paths_per_method[CVAE_KEY].append(generated_log_path)

    gen_end_time = time.time()

    gen_time = gen_end_time - gen_start_time

    with open(os.path.join(output_path, 'generation_time.txt'), 'w') as f:
      f.write(str(gen_time))
  else:
    log_paths_per_method[CVAE_KEY] = glob.glob(os.path.join(output_path, 'gen') + '/*.csv')

  # LOG_4
  if should_use_log_3:
    log_paths_per_method[LOG_3_KEY] = glob.glob(os.path.join(input_path, 'log_3') + '/*.csv')
    assert len(log_paths_per_method[LOG_3_KEY]) == 3
  
  # LSTM_1
  if should_use_lstm_1:
    log_paths_per_method[LSTM_1_KEY] = glob.glob(os.path.join(input_path, 'lstm_1') + '/*.csv')
    if should_use_cvae:
      assert len(log_paths_per_method[CVAE_KEY]) == len(log_paths_per_method[LSTM_1_KEY])

  # LSTM_2
  if should_use_lstm_2:
    log_paths_per_method[LSTM_2_KEY] = glob.glob(os.path.join(input_path, 'lstm_2') + '/*.csv')
    if should_use_cvae:
      assert len(log_paths_per_method[CVAE_KEY]) == len(log_paths_per_method[LSTM_2_KEY])

  # Transformer
  if should_use_transformer_1:
    log_paths_per_method[TRANSFORMER_1_KEY] = glob.glob(os.path.join(input_path, 'transformer_1') + '/*.csv')
    if should_use_cvae:
      assert len(log_paths_per_method[CVAE_KEY]) == len(log_paths_per_method[TRANSFORMER_1_KEY])

  # Transformer_2
  if should_use_transformer_2:
    log_paths_per_method[TRANSFORMER_2_KEY] = glob.glob(os.path.join(input_path, 'transformer_2') + '/*.csv')
    if should_use_cvae:
      assert len(log_paths_per_method[CVAE_KEY]) == len(log_paths_per_method[TRANSFORMER_2_KEY])

  # ProcessGAN
  # It generates a log of the size of the train log. So, given a 60/20/20 split, we expect
  # to have 3 generated logs to compare to the test log
  if should_use_processgan_1:
    log_paths_per_method[PROCESSGAN_1_KEY] = glob.glob(os.path.join(input_path, 'processgan_1') + '/*.csv')
    assert len(log_paths_per_method[PROCESSGAN_1_KEY]) == 3 or len(log_paths_per_method[PROCESSGAN_1_KEY]) == 4

  # Recompute labels on generated data
  if should_recompute_labels_on_generated_data:
    if not os.path.exists(os.path.join(output_path, 'recomputed-labels')):
      os.makedirs(os.path.join(output_path, 'recomputed-labels'))

    for method in DEFAULT_METHODS_DICT.keys():
      if method in [LOG_3_KEY, LSTM_1_KEY, PROCESSGAN_1_KEY, TRANSFORMER_1_KEY]: continue

      print(f'Recomputing labels on generated data with {method}...')

      recomputed_labels_results = recompute_labels_on_generated_data(list(generation_config['LABELS'].keys()), log_paths_per_method[method], log_name=log_name)
      save_dict_to_json(recomputed_labels_results, filepath=os.path.join(output_path, 'recomputed-labels', f'recomputed-labels-{method}.json'))

  # Compute conformance and log distance measures
  log_distance_measures = {
    'full': {}, # conformance and log distance measures computed on full log
    **({'deviant': {}} if 'deviant' in log_distance_measures_also_compute_filtered_by else {}),
    **({'regular': {}} if 'regular' in log_distance_measures_also_compute_filtered_by else {}),
  }

  for filter_log_by in [None] + log_distance_measures_also_compute_filtered_by:
    filter_log_by_idx = 'full' if filter_log_by == None else filter_log_by
  
    # Compute conformance with declare
    if should_compute_conformance and not should_skip_all_metrics_computation:
      print(f'Computing conformance for log filtered by {filter_log_by}...')

      conformances = copy.deepcopy(DEFAULT_METHODS_DICT)

      declare_model_path = discover_declare_model(
        dataset_info['FULL'],
        log_csv_separator=dataset_info['CSV_SEPARATOR'],
        case_key=dataset_info['TRACE_KEY'],
        activity_key=dataset_info['ACTIVITY_KEY'],
        timestamp_key='timestamp',
        output_path=os.path.join(output_path, 'conformance'),
        consider_vacuity=consider_vacuity,
        min_support=min_support,
        itemsets_support=itemsets_support,
        max_declare_cardinality=max_declare_cardinality,
        filter_log_by=filter_log_by,
      )

      for method, log_paths in log_paths_per_method.items():
        # we cannot compute filtered metrics for methods that do not generate labels
        if filter_log_by is not None and method in [LSTM_1_KEY, TRANSFORMER_1_KEY, PROCESSGAN_1_KEY]: continue

        for i, log_path in enumerate(log_paths):
          try:
            conformance = compute_conformance(
              log_path,
              declare_model_path,
              output_path=os.path.join(output_path, 'conformance'),
              output_name=f'gen_{method}_{filter_log_by_idx}_{i+1}',
              dataset_info=dataset_info,
              filter_log_by=filter_log_by,
              consider_vacuity=consider_vacuity
            )
          except Exception as e:
            print(f'Error computing conformance for {method} {i+1}: {e}\nTrying with .xes log...')
            # If an error occurs, try the .xes log instead
            log_path = log_path.replace('.csv', '.xes')
            conformance = compute_conformance(
              log_path,
              declare_model_path,
              output_path=os.path.join(output_path, 'conformance'),
              output_name=f'gen_{method}_{filter_log_by_idx}_{i+1}',
              dataset_info=dataset_info,
              filter_log_by=filter_log_by,
              consider_vacuity=consider_vacuity
            )

          conformances[method].append(conformance)

      log_distance_measures[filter_log_by_idx]['conformance'] = conformances
      save_dict_to_json(conformances, filepath=os.path.join(output_path, f'conformance-{filter_log_by_idx}.json'))

    # Compute log distance measures
    for measure_to_compute in log_distance_measures_to_compute:
      print(f'Computing {measure_to_compute} for log filtered by {filter_log_by}...')

      measure_results = copy.deepcopy(DEFAULT_METHODS_DICT)

      for method, log_paths in log_paths_per_method.items():
        # we cannot compute filtered metrics for methods that do not generate labels
        if filter_log_by is not None and method in [LSTM_1_KEY, TRANSFORMER_1_KEY, PROCESSGAN_1_KEY]: continue

        # we cannot compute cwd for methods that do not generate resources/roles
        if measure_to_compute == 'cwd' and method not in [LOG_3_KEY, CVAE_KEY, LSTM_1_KEY, LSTM_2_KEY]: continue

        for i, log_path in enumerate(log_paths):
          measure_results[method].append(
            compute_log_distance_measure(
              dataset_info['TEST'],
              log_path,
              dataset_info,
              measure=measure_to_compute,
              method=method,
              filter_log_by=filter_log_by,
              cwd_resource_to_role_mapping_file=cwd_resource_to_role_mapping_file,
              gen_log_trace_key=dataset_info['TRACE_KEY'] if method in [LOG_3_KEY] else 'case:concept:name',
              gen_log_activity_key=dataset_info['ACTIVITY_KEY'] if method in [LOG_3_KEY] else 'concept:name',
              gen_resource_key=dataset_info['RESOURCE_KEY'] if method in [LOG_3_KEY] else 'org:resource',
            )
          )

      log_distance_measures[filter_log_by_idx][measure_to_compute] = measure_results
      save_dict_to_json(measure_results, filepath=os.path.join(output_path, f'{measure_to_compute}-{filter_log_by_idx}.json'))

    # Save log_distance_measures
    log_distance_measures_path = os.path.join(output_path, f'log_distance_measures-{filter_log_by_idx}.json')
    if not should_skip_all_metrics_computation:
      save_dict_to_json(log_distance_measures[filter_log_by_idx], filepath=log_distance_measures_path)

    # Plot boxplots
    if should_plot_boxplots:
      print('Plotting boxplots...')

      # split dict in two dicts by keys
      cf_log_distance_measures = {
        key: value for key, value in log_distance_measures[filter_log_by_idx].items() if key in ['conformance', 'cfld', 'ngram_2', 'ngram_3', 'ngram_4', 'ngram_5']
      }
      ts_log_distance_measures = {
        key: value for key, value in log_distance_measures[filter_log_by_idx].items() if key in ['aed', 'cad', 'ced', 'red', 'ctd']
      }
      res_log_distance_measures = {
        key: value for key, value in log_distance_measures[filter_log_by_idx].items() if key in ['cwd']
      }
      plot_boxplots(log_distance_measures[filter_log_by_idx], output_path=output_path, output_filename=f'log_distance_measures-{filter_log_by_idx}.png')
      if cf_log_distance_measures:
        plot_boxplots(cf_log_distance_measures, output_path=output_path, output_filename=f'cf_log_distance_measures-{filter_log_by_idx}.png')
      if ts_log_distance_measures:
        plot_boxplots(ts_log_distance_measures, output_path=output_path, output_filename=f'ts_log_distance_measures-{filter_log_by_idx}.png')
      if res_log_distance_measures:
        plot_boxplots(res_log_distance_measures, output_path=output_path, output_filename=f'res_log_distance_measures-{filter_log_by_idx}.png')

  # Plot t-sne
  if should_plot_tsne:
    print('Plotting t-SNE...')

    for method, log_paths in log_paths_per_method.items():
      if method in [LOG_3_KEY, LSTM_1_KEY, LSTM_2_KEY, TRANSFORMER_1_KEY, TRANSFORMER_2_KEY, PROCESSGAN_1_KEY]: continue

      for i, generated_log_path in enumerate(log_paths):
        for consider_timestamps in [True, False]:
          tsne_filename_ts = 'ts' if consider_timestamps else 'no-ts'

          plot_tsne(
            dataset_info['TEST'],
            generated_log_path,
            consider_timestamps=consider_timestamps,
            output_path=os.path.join(output_path, 'tsne'),
            plot_filename=f'tsne-plot-gen-{method}-{tsne_filename_ts}-{i+1}.png',
            plot_title=f'T-SNE Gen {method} {i+1}',
            max_og_0=tsne_max_og_0,
            max_og_1=tsne_max_og_1,
            max_gen_0=tsne_max_gen_0,
            max_gen_1=tsne_max_gen_1,
            dataset_info=dataset_info,
            seed=seed,
          )

  # Plot distribution of trace lengths
  if should_plot_trace_length_distribution:
    print('Plotting trace length distributions...')

    for method, log_paths in log_paths_per_method.items():
      if method in [LOG_3_KEY]: continue
      
      for i, generated_log_path in enumerate(log_paths):
        plot_trace_length_distribution(
          dataset_info['TEST'],
          generated_log_path,
          dataset_info=dataset_info,
          generated_log_trace_key='case:concept:name',
          output_path=os.path.join(output_path, 'trace-length-distributions'),
          output_filename=f'trace-length-distribution-gen-{method}-{i+1}.png',
        )

  # Print variant statistics
  if should_compute_variant_stats:
    print('Computing variant stats...')

    variant_stats_output_path = os.path.join(output_path, 'variant-stats')
    if not os.path.exists(variant_stats_output_path):
      os.makedirs(variant_stats_output_path)
    
    variant_stats_gen = copy.deepcopy(DEFAULT_METHODS_DICT)

    for method, generated_log_paths in log_paths_per_method.items():
      if method in [LOG_3_KEY]: continue

      for i, generated_log_path in enumerate(generated_log_paths):
        variant_stats_gen[method].append(
          compute_variant_stats(
            dataset_info['TRAIN'],
            dataset_info['TEST'],
            generated_log_path,
            dataset_info=dataset_info
          )
        )
        save_dict_to_json(variant_stats_gen[method][i], filepath=os.path.join(variant_stats_output_path, f'variant-stats-gen-{method}-{i+1}.json'))

    # Compute average variant stats
    variant_stats_avg = copy.deepcopy(DEFAULT_METHODS_DICT)
    for method, variant_stats in variant_stats_gen.items():
      if method in [LOG_3_KEY]: continue

      variant_stats_avg[method] = {
        'num_generated_variants': sum([stats['num_generated_variants'] for stats in variant_stats]) / len(variant_stats),
        'num_shared_variants_train': sum([stats['num_shared_variants_train'] for stats in variant_stats]) / len(variant_stats),
        'num_shared_variants_test': sum([stats['num_shared_variants_test'] for stats in variant_stats]) / len(variant_stats),
      }
    
    save_dict_to_json(variant_stats_avg, filepath=os.path.join(variant_stats_output_path, 'variant-stats-AVERAGE.json'))


  # Plot event duration distributions
  if should_plot_activity_duration_distributions:
    print('Plotting activity duration distributions...')

    for method, log_paths in log_paths_per_method.items():
      if method in [LOG_3_KEY]: continue

      for i, generated_log_path in enumerate(log_paths):
        plot_activity_duration_distributions(
          [dataset_info['TEST'], generated_log_path],
          dataset_info,
          output_path=os.path.join(output_path, 'activity-duration-distributions'),
          output_filename=f'activity-duration-distributions-gen-{method}-{i+1}.png',
          activity_duration_distributions_filter_by_label=activity_duration_distributions_filter_by_label,
          case_col_name=[dataset_info['TRACE_KEY'], 'case:concept:name'],
          timestamp_col_name=['time:timestamp', 'time:timestamp'],
          activity_col_name=[dataset_info['ACTIVITY_KEY'], 'concept:name'],
        )

  # Plot resource distribution
  if should_plot_resource_distribution:
    print('Plotting resource distributions...')

    for method, log_paths in log_paths_per_method.items():
      if method in [LOG_3_KEY, LSTM_1_KEY, LSTM_2_KEY, TRANSFORMER_1_KEY, TRANSFORMER_2_KEY, PROCESSGAN_1_KEY]: continue
    
      for i, generated_log_path in enumerate(log_paths):
        plot_resource_distribution(
          dataset_info['TEST'],
          generated_log_path,
          dataset_info=dataset_info,
          output_path=os.path.join(output_path, 'resource-distributions'),
          output_filename=f'resource-distribution-gen-{method}-{i+1}.png',
        )

  if should_plot_activity_by_resource_distribution:
    print('Plotting activity by resource distributions...')

    for method, log_paths in log_paths_per_method.items():
      if method in [LOG_3_KEY, LSTM_1_KEY, LSTM_2_KEY, TRANSFORMER_1_KEY, TRANSFORMER_2_KEY, PROCESSGAN_1_KEY]: continue

      for i, generated_log_path in enumerate(log_paths):
        plot_activity_distribution_by_resource(
          dataset_info['TEST'],
          generated_log_path,
          dataset_info=dataset_info,
          output_path=os.path.join(output_path, 'activity-by-resource-distributions'),
          output_filename=f'activity-by-resource-distribution-gen-{method}-{i+1}.png',
        )

  # Plot trace attribute distributions
  if should_plot_trace_attribute_distributions:
    print('Plotting trace attribute distributions...')
      
    for i, generated_log_path in enumerate(log_paths_per_method[CVAE_KEY]):
      plot_trace_attribute_distributions(
        dataset_info['TEST'],
        generated_log_path,
        output_path=os.path.join(output_path, 'trace-attribute-distributions'),
        output_filename=f'trace-attribute-distributions-gen-VAE-{i+1}.png',
        trace_attributes=trace_attributes,
        original_log_trace_key=dataset_info['TRACE_KEY'],
        generated_log_trace_key='case:concept:name',
      )
