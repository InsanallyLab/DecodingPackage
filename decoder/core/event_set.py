class EventSet:
    def __init__(self, name, possible_values, metadata=None):
        self.name = name
        self.possible_values = possible_values
        self.metadata = metadata if metadata else {}
        self.events = {}  # Dictionary to store events with labels

    def add_event(self, timepoint, label):
        # Check if label is among possible values
        if label not in self.possible_values:
            raise ValueError(f"Label {label} not in possible values.")
        
        # Check if timepoint is unique
        if timepoint in self.events:
            raise ValueError(f"Timepoint {timepoint} already exists.")
        
        # Add the event
        self.events[timepoint] = label

    def get_events(self):
        return self.events
