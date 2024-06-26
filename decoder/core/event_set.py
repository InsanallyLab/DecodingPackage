# class EventSet:
#     """Represents a collection of events in a session.

#     Attributes:
#         possible_values (set): Set of possible labels/values for events.
#         metadata (dict): Additional information about the event set.
#         events (dict): Dictionary of events, with timestamps as keys and labels as values.
#     """

#     def __init__(self, name, timestamps, labels, possible_values=None, metadata=None):
#         """
#         Initialize an EventSet.

#         Args:
#             name (str): name of the event set. typically type of event e.g. lick. 
#             timestamps (list): List of timestamps for events.
#             labels (list): List of labels corresponding to the timestamps.
#             possible_values (list, optional): List of possible labels/values for events. Defaults to None.
#             metadata (dict, optional): Additional information about the event set. Defaults to None.

#         Raises:
#             ValueError: If the length of timestamps and labels doesn't match or if any label is not in possible_values.
#         """
#         self.name = name

#         if len(timestamps) != len(labels):
#             raise ValueError("Length of timestamps and labels must match.")

#         if possible_values and not all(label in possible_values for label in labels):
#             raise ValueError("Some labels are not in the possible values list.")

#         # CHANGED: possible_values stored as a set to improve time complexity of checking if a value is in possible_values
#         self.possible_values = possible_values if possible_values else set(labels)
#         self.metadata = metadata if metadata else {}
#         self.events = dict(zip(timestamps, labels))

#     def add_event(self, timepoint, label):
#         """Add a new event to the collection.

#         Args:
#             timepoint (float): The timestamp of the event.
#             label (str): The label of the event.

#         Raises:
#             ValueError: If the label is not allowed or the timestamp already exists.
#         """
#         if label not in self.possible_values:
#             raise ValueError(f"Label {label} not in possible values.")
#         if timepoint in self.events:
#             raise ValueError(f"Timepoint {timepoint} already exists.")
#         self.events[timepoint] = label

#     def delete_event(self, timepoint):
#         """Remove an event based on its timestamp.

#         Args:
#             timepoint (float): The timestamp of the event to be removed.

#         Raises:
#             KeyError: If the timepoint does not exist.
#         """
#         if timepoint not in self.events:
#             raise KeyError(f"Timepoint {timepoint} does not exist.")
#         del self.events[timepoint]

#     def get_sorted_timestamps(self):
#         """Return a sorted list of timestamps of events.

#         Returns:
#             list: Sorted list of event timestamps.
#         """
#         return sorted(self.events.keys())
