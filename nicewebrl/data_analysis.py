"""

These are some basic functions to help with data analysis.
"""

from datetime import datetime

def time_diff(t1, t2) -> float:
  # Convert string timestamps to datetime objects
  t1 = datetime.strptime(t1, "%Y-%m-%dT%H:%M:%S.%fZ")
  t2 = datetime.strptime(t2, "%Y-%m-%dT%H:%M:%S.%fZ")

  # Calculate the time difference
  time_difference = t2 - t1

  # Convert the time difference to milliseconds
  return time_difference.total_seconds() * 1000


def compute_reaction_time(datum) -> float:
  # Calculate the time difference
  return time_diff(datum["image_seen_time"], datum["action_taken_time"])


