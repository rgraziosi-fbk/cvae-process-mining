from statistics import mean

import pandas as pd
from scipy.stats import wasserstein_distance

from log_distance_measures.config import EventLogIDs, DistanceMetric
from log_distance_measures.earth_movers_distance import earth_movers_distance
from log_distance_measures.config import DEFAULT_CSV_IDS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def circadian_workforce_distribution_distance1(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of the workforce by hour (number of observed active workers
    in each hour) in two event logs, windowed by weekday (e.g., the workforces measured for each Monday in a log are
    aggregated as an average into one single Monday).

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the timestamp distribution of the two event logs windowed by weekday.
    """
    # Get discretized start and/or end timestamps
    original_discrete_events = _discretize1(original_log, original_ids)
    simulated_discrete_events = _discretize1(simulated_log, simulated_ids)
    # Compute the distance between the instant in the event logs for each weekday
    distances = []
    for week_day in range(7):  # All weekdays
        original_window = original_discrete_events[original_discrete_events['weekday'] == week_day]
        simulated_window = simulated_discrete_events[simulated_discrete_events['weekday'] == week_day]
        if len(original_window) > 0 and len(simulated_window) > 0:
            # Compute 2-D dict with each bin (hour) and its number of observations
            original_2d = original_window.drop('weekday', axis=1).set_index('hour')['workforce'].to_dict()
            simulated_2d = simulated_window.drop('weekday', axis=1).set_index('hour')['workforce'].to_dict()
            # Both have observations in this weekday
            if metric == DistanceMetric.EMD:
                distances += [earth_movers_distance(original_2d, simulated_2d) / sum(original_window['workforce'])]
            else:
                # Transform to 1D array
                original_1d = [element for key in original_2d for element in [key] * int(original_2d[key] * 100)]
                simulated_1d = [element for key in simulated_2d for element in [key] * int(simulated_2d[key] * 100)]
                # Measure 1-WD

                # Create histograms
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.hist(original_1d, bins=len(original_2d), alpha=0.7, color='blue', edgecolor='black')
                plt.title("Original Distribution")
                plt.xlabel("Value")
                plt.ylabel("Frequency")

                plt.subplot(1, 2, 2)
                plt.hist(simulated_1d, bins=len(simulated_2d), alpha=0.7, color='red', edgecolor='black')
                plt.title("Simulated Distribution")
                plt.xlabel("Value")
                plt.ylabel("Frequency")

                plt.tight_layout()
                plt.show()

                distances += [wasserstein_distance(original_1d, simulated_1d)]
        elif len(original_window) == 0 and len(simulated_window) == 0:
            # Both have no observations in this weekday
            distances += [0.0]
        else:
            # Only one has observations in this weekday, penalize with max distance value
            distances += [23.0]  # 23 is the maximum value for two histograms with values between 0 and 23.
    # Compute distance metric
    print(distances)
    distance = mean(distances)
    if normalize:
        distance = distance / 23.0
    # Return metric
    return distance

def circadian_workforce_distribution_distance_weekly(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of the workforce by hour (number of observed active workers
    in each hour) in two event logs, windowed by weekday (e.g., the workforces measured for each Monday in a log are
    aggregated as an average into one single Monday).

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the timestamp distribution of the two event logs windowed by weekday.
    """
    # Get discretized start and/or end timestamps
    original_discrete_events = _discretize1(original_log, original_ids).drop('hour', axis=1)
    original_discrete_events = [int(x) for x in original_discrete_events['workforce'].tolist()]
    # print(original_discrete_events)

    simulated_discrete_events = _discretize1(simulated_log, simulated_ids).drop('hour', axis=1)
    simulated_discrete_events = [int(x) for x in simulated_discrete_events['workforce'].tolist()]
    # print(simulated_discrete_events)

    original_discrete_events = np.array(original_discrete_events) / max(original_discrete_events)
    simulated_discrete_events = np.array(simulated_discrete_events) / max(simulated_discrete_events)

    # Days of the week
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Optional: show plots
    # # Create subplots
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # # Plot Test Set
    # axes[0].bar(days, original_discrete_events, color='blue', edgecolor='black', alpha=0.7)
    # axes[0].set_xlabel("Day of the Week")
    # axes[0].set_ylabel("Frequency")
    # axes[0].set_title("Test Set")
    # axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # # Plot VAE
    # axes[1].bar(days, simulated_discrete_events, color='red', edgecolor='black', alpha=0.7)
    # axes[1].set_xlabel("Day of the Week")
    # axes[1].set_title("LSTM_1")
    # axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # # Show the plots
    # plt.tight_layout()
    # plt.show()

    distances = []

    distances += [wasserstein_distance(original_discrete_events, simulated_discrete_events)]
    # Compute distance metric
    distance = mean(distances)

    # Return metric
    return distance


def _discretize1(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
) -> pd.DataFrame:
    """
    Create a pd.Dataframe storing, for each hour (0-23) of each weekday, the average number of different resources
    that were actively working (an event associated to them was registered in that hour) through the entire event log.

    :param event_log: event log to measure the workforces of.
    :param log_ids: mapping for the column IDs of the event log.

    :return: A pd.Dataframe with the average active workforce for each hour for each day of the week.
    """
    # Get the instants to discretize
    # log_ids.start_time = "time:timestamp"
    # log_ids.resource = "resource"
    # log_ids.end_time = "time:timestamp"
    start_times = event_log[[log_ids.start_time, log_ids.resource]].rename(columns={log_ids.start_time: 'instant'})
    end_times = event_log[[log_ids.end_time, log_ids.resource]].rename(columns={log_ids.end_time: 'instant'})
    discretized = pd.concat([start_times, end_times]).reset_index(drop=True)
    # Add their weekday, hour and combination day-hour
    discretized['weekday'] = discretized['instant'].apply(lambda instant: instant.day_of_week)
    discretized['hour'] = discretized['instant'].apply(lambda instant: instant.hour)
    discretized['day-hour'] = discretized['instant'].apply(lambda instant: instant.strftime(format="%Y-%m-%d %H"))
    # Compute observed number of Mondays, Tuesdays...
    discretized['day'] = discretized['instant'].apply(lambda instant: instant.strftime(format="%Y-%m-%d"))
    days = discretized[['day', 'weekday']].drop_duplicates().groupby('weekday').size().to_dict()
    # Keep unique resources per day-hour
    discretized.drop_duplicates(subset=['day-hour', log_ids.resource], inplace=True)
    # Aggregate number of resources per day of the week and hour, and compute avg
    discretized = discretized.groupby(['weekday', 'hour']).size().reset_index().rename(columns={0: 'workforce'})
    discretized['workforce'] = discretized.apply(lambda row: row['workforce'] / days[row['weekday']], axis=1)
    # Return dataframe with weekday, hour, and average resources
    return discretized


if __name__ == '__main__':
    event_log_ids = EventLogIDs(  # These values are stored in DEFAULT_CSV_IDS
            case="Case ID",
            activity="Activity",
            start_time="time:timestamp",
            end_time="time:timestamp",
            resource="resource"
        )


    original_log = pd.read_csv("/Users/francescameneghello/Downloads/traffic_fines_TEST.csv", sep=";")
    original_log[event_log_ids.start_time] = pd.to_datetime(original_log[event_log_ids.start_time], utc=True)
    original_log[event_log_ids.end_time] = pd.to_datetime(original_log[event_log_ids.end_time], utc=True)

    simulated_log = pd.read_csv("/Users/francescameneghello/Downloads/gen1.csv", sep=";")
    simulated_log[event_log_ids.start_time] = pd.to_datetime(simulated_log[event_log_ids.start_time], utc=True)
    simulated_log[event_log_ids.end_time] = simulated_log[event_log_ids.end_time].dt.date
    simulated_log[event_log_ids.end_time] = pd.to_datetime(simulated_log[event_log_ids.end_time], utc=True)


    distance = circadian_workforce_distribution_distance_weekly(
        original_log, EventLogIDs,  # First event log and its column id mappings
        simulated_log, EventLogIDs,  # Second event log and its column id mappings
    )
