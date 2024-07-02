import pynapple as nap

def convert_units(nap_object, input_units: str = "s"):
    """
    Reinitializes all Pynapple input objects with the correct time units.
    Pynapple assumes that all input data is passed in as seconds. If this
    is not the case, use this function to convert all Pynapple objects 
    to the correct time unit.

    Parameters
    ----------
    nap_object : Ts, Tsd, or IntervalSet 
        Pynapple object to be converted to the correct time unit.
    units : str, optional
        ('us', 'ms', 's' [default])

    Returns 
    -------
    The Pynapple object converted to the correct time unit.
    """
    if isinstance(nap_object, nap.core.time_series.Ts):
        return nap.Ts(t=nap_object.t, time_units=input_units)

    if isinstance(nap_object, nap.core.time_series.Tsd):
        return nap.Tsd(t=nap_object.t, d=nap_object.d, time_units=input_units)
    
    if isinstance(nap_object, nap.core.interval_set.IntervalSet):
        return nap.IntervalSet(start=nap_object.start, end=nap_object.end, time_units=input_units)